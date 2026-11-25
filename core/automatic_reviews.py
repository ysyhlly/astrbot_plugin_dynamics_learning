"""Batch compatible automatic first reviews with a durable wait and retry queue."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass

from .. import series_panels
from .contract import MODEL_INPUT_LIMIT, MalformedModelResponse, encode, review_result
from .model_projection import REVIEW_BATCH_PAYLOAD_BUDGET, build_batch_review_payload, utf8_len
from .review_batches import ReviewDeferred, ReviewUnavailable
from .store import Conflict

MAX_ITEMS = 4
WAIT_SECONDS = 300
DEFAULT_COALESCE_SECONDS = 15.0
BUDGET_BACKOFF_STEPS = (60.0, 120.0, 300.0)


@dataclass(frozen=True)
class ReviewDispatch:
    scope: str
    proposals: tuple[str, ...]
    kind: str
    provider_id: str
    model: str
    provider_instance: object | None
    operation_id: str

    @classmethod
    def for_items(cls, items, kind, operation_id=""):
        item = items[0]
        route = item.context
        return cls(
            item.proposal["scope"],
            tuple(i.proposal["id"] for i in items),
            kind,
            route.provider_id,
            route.model,
            route.provider_instance,
            operation_id or uuid.uuid4().hex,
        )


def urgent_review(p):
    return (
        p["category"] in {"expression_adjustment", "scene_adjustment"}
        or p["source"].startswith("human:")
        or any(e.get("kind") == "operator_correction" for e in p["evidence"])
    )


def first_review(p):
    # Expression variants are reviewed during winner selection.
    return (
        p["state"] == "pending"
        and not p["reviews"]
        and p["mode"] in {"llm", "llm_then_human"}
        and p["category"] != "expression_style"
        and not urgent_review(p)
    )


def batch_verdicts(value, expected):
    valid = {}
    seen = set()
    if not isinstance(value, dict) or not isinstance(value.get("verdicts"), list):
        return valid
    for item in value["verdicts"]:
        identity = item.get("proposal_id") if isinstance(item, dict) else None
        if not isinstance(identity, str) or identity not in expected:
            continue
        if identity in seen:
            valid.pop(identity, None)
            continue
        seen.add(identity)
        try:
            valid[identity] = review_result({k: v for k, v in item.items() if k != "proposal_id"})
        except (TypeError, ValueError):
            continue
    return valid


def batch_payload(items):
    first = items[0].proposal
    entries = [
        (
            item.proposal,
            item.context.target_material
            if item.proposal["category"] in {"expression_adjustment", "scene_adjustment"}
            else ...,
        )
        for item in items
    ]
    return build_batch_review_payload(first["scope"], first["persona"], entries)


class AutomaticReviews:
    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store
        self.queue = self.store.automatic_reviews
        self.wake = asyncio.Event()
        self.active = {}
        self.running = False
        self.reconcile = True
        self.wait_seconds = WAIT_SECONDS
        self._budget_backoff_until = 0.0
        self._budget_backoff_streak = 0
        self._provider_backoff_until: dict[str, float] = {}
        self._provider_backoff_streak: dict[str, int] = {}

    def _resolve_provider_key(self, *, role: str = "review_provider", provider_id: str | None = None) -> str:
        if provider_id is not None and str(provider_id).strip():
            return str(provider_id).strip()
        pid = self.engine.provider_for(role) if hasattr(self.engine, "provider_for") else ""
        return str(pid or "").strip() or role

    def _provider_circuit_remaining(self, *, role: str = "review_provider", provider_id: str | None = None) -> float:
        ctx = getattr(self.engine, "_host_context", None)
        if ctx is None:
            return 0.0
        pid = (
            str(provider_id).strip()
            if provider_id is not None and str(provider_id).strip()
            else (self.engine.provider_for(role) if hasattr(self.engine, "provider_for") else "")
        )
        if not pid:
            return 0.0
        probe = getattr(series_panels, "provider_circuit_remaining", lambda _c, _p: 0.0)
        try:
            return max(0.0, float(probe(ctx, pid) or 0.0))
        except Exception:
            return 0.0

    def note_budget_busy(
        self,
        exc_or_retry_after=None,
        *,
        now: float | None = None,
        role: str = "review_provider",
        provider_id: str | None = None,
    ) -> float:
        stamp = time.time() if now is None else float(now)
        if isinstance(exc_or_retry_after, BaseException):
            retry_after = max(0.0, float(getattr(exc_or_retry_after, "retry_after", 0.0) or 0.0))
        else:
            retry_after = max(0.0, float(exc_or_retry_after or 0.0))
        key = self._resolve_provider_key(role=role, provider_id=provider_id)
        streak = self._provider_backoff_streak.get(key, 0)
        step = BUDGET_BACKOFF_STEPS[min(streak, len(BUDGET_BACKOFF_STEPS) - 1)]
        wait = max(step, retry_after)
        self._provider_backoff_streak[key] = streak + 1
        until = max(self._provider_backoff_until.get(key, 0.0), stamp + wait)
        self._provider_backoff_until[key] = until
        if key == self._resolve_provider_key(role="review_provider"):
            self._budget_backoff_streak = self._provider_backoff_streak[key]
            self._budget_backoff_until = max(self._budget_backoff_until, until)
        return until

    def reset_budget_backoff(self, *, role: str | None = None, provider_id: str | None = None) -> None:
        if role is None and provider_id is None:
            self._provider_backoff_streak.clear()
            self._provider_backoff_until.clear()
            self._budget_backoff_streak = 0
            self._budget_backoff_until = 0.0
            return
        key = self._resolve_provider_key(role=role or "review_provider", provider_id=provider_id)
        self._provider_backoff_streak.pop(key, None)
        self._provider_backoff_until.pop(key, None)
        if key == self._resolve_provider_key(role="review_provider"):
            self._budget_backoff_streak = 0
            self._budget_backoff_until = 0.0

    def budget_backoff_remaining(
        self,
        *,
        now: float | None = None,
        role: str = "review_provider",
        provider_id: str | None = None,
    ) -> float:
        stamp = time.time() if now is None else float(now)
        key = self._resolve_provider_key(role=role, provider_id=provider_id)
        until = self._provider_backoff_until.get(key, 0.0)
        if provider_id is None and key == self._resolve_provider_key(role="review_provider"):
            until = max(until, self._budget_backoff_until)
        circuit_rem = self._provider_circuit_remaining(role=role, provider_id=provider_id)
        if circuit_rem > 0:
            until = max(until, stamp + circuit_rem)
            self._provider_backoff_until[key] = until
            if key == self._resolve_provider_key(role="review_provider"):
                self._budget_backoff_until = max(self._budget_backoff_until, until)
        return max(0.0, until - stamp)

    def is_budget_backed_off(
        self,
        *,
        now: float | None = None,
        role: str = "review_provider",
        provider_id: str | None = None,
    ) -> bool:
        return self.budget_backoff_remaining(now=now, role=role, provider_id=provider_id) > 0.0

    @property
    def enabled(self):
        return (
            self.running
            and self.engine.config.get("enabled", True)
            and self.engine.config.get("automatic_review_batch_enabled", True)
        )

    def candidate(self, p):
        try:
            mode = self.engine.proposal_mode(p) if hasattr(self.engine, "proposal_mode") else self.engine.mode(p["category"])
        except ValueError:
            return False
        return (
            first_review(p)
            and self.engine.allowed(p["scope"])
            and self.engine.current_personas.get(p["scope"], p["persona"]) == p["persona"]
            and mode in {"llm", "llm_then_human"}
            and p["created"] + p["days"] * 86400 > time.time()
            and self.store.review_claim(p["id"]) is None
        )

    def notify_context_change(self):
        self.reconcile = True
        self.wake.set()

    def enqueue_candidates(self, proposals, now):
        candidates = [p for p in proposals if self.candidate(p)]
        return self.queue.enqueue(candidates, now) if candidates else set()

    async def enqueue(self, proposals):
        if not self.enabled:
            return set()
        accepted = await self.engine.review_write(lambda: self.enqueue_candidates(proposals, time.time()))
        if accepted:
            self.wake.set()
        return accepted

    def item(self, p):
        if not self.enabled or not self.candidate(p) or ("review", p["id"]) in self.engine.operation_locks:
            return None
        item = self.engine.review_item(p)
        if self.engine._provider_tracks_reviews and item.context.provider_instance is None:
            return None
        return item if item.context.automatic_allowed else None

    def coalesce_seconds(self):
        raw = self.engine.config.get("automatic_review_coalesce_seconds", DEFAULT_COALESCE_SECONDS)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = DEFAULT_COALESCE_SECONDS
        return max(0.0, value)

    def ready(self, now, busy, *, discover=True):
        if discover:
            self.enqueue_candidates(self.queue.undiscovered(self.engine.effective_sessions(), now), now)

        groups = defaultdict(list)
        next_at = float("inf")
        for row, p in self.queue.waiting(now, include_leased=True):
            if p["id"] in busy:
                continue
            available = max(row["retry"], row["lease_until"])
            if available > now:
                next_at = min(next_at, available)
                continue
            item = self.item(p)
            if item is not None:
                key = (item.group_key(), p["id"] if row["singleton"] else "")
                groups[key].append((row, item))

        coalesce = self.coalesce_seconds()
        batch_enabled = bool(self.engine.config.get("automatic_review_batch_enabled", True))
        ready = []
        for items in groups.values():
            row, oldest = items[0]
            selected = [(row, oldest)]
            entries = {oldest.proposal["entry_id"]}
            full = bool(row["singleton"]) or not batch_enabled
            for candidate in items[1:]:
                entry = candidate[1].proposal["entry_id"]
                if entry in entries:
                    continue
                candidate_payload = batch_payload([item for _, item in selected] + [candidate[1]])
                if (
                    len(encode(candidate_payload)) > MODEL_INPUT_LIMIT
                    or utf8_len(candidate_payload) > REVIEW_BATCH_PAYLOAD_BUDGET
                ):
                    full = True
                    break
                selected.append(candidate)
                entries.add(entry)
                if len(selected) == MAX_ITEMS:
                    full = True
                    break
            has_urgent = any(urgent_review(item.proposal) for _, item in selected)
            max_wait_deadline = row["ready"] + self.wait_seconds
            newest_ready = max(r["ready"] for r, _ in selected)
            deadline = min(max_wait_deadline, newest_ready + coalesce)
            if full or has_urgent or deadline <= now:
                ready.append((deadline, selected))
            else:
                next_at = min(next_at, deadline)
        ready.sort(key=lambda group: group[0])
        return [items for _, items in ready], next_at

    def prepare(self, rows, lease):
        now = time.time()
        # Waiting never takes a manual claim. Reserve only at the model slot.
        with self.store.transaction():
            selected = []
            for row, previous in rows:
                try:
                    p = self.store.get(row["proposal"])
                except ValueError:
                    continue
                if p["updated"] != row["updated"]:
                    continue
                item = self.item(p)
                if item is None or item.group_key() != previous.group_key():
                    continue
                # The production provider timeout is capped at 180 seconds.
                if self.queue.claim(p["id"], p["updated"], lease, now, now + 200):
                    selected.append(item)
            return selected

    async def dispatch(self, rows):
        lease = uuid.uuid4().hex
        sent = []
        retry = 0

        async def prepare():
            async with self.engine.lock:
                sent[:] = await self.engine.review_write(lambda: self.prepare(rows, lease))
            if not sent:
                raise ReviewUnavailable("automatic review invalidated before dispatch")
            if len(sent) > 1:
                dispatch = ReviewDispatch.for_items(sent, "automatic_batch", lease)
                return sent[0].single_instruction(), batch_payload(sent), dispatch
            kind = "automatic_item_retry" if rows[0][0]["singleton"] else "automatic_first"
            return sent[0].single_instruction(), sent[0].data(), ReviewDispatch.for_items(sent, kind, lease)

        try:
            try:
                raw = await self.engine.invoke_model(
                    "review_provider", "", {"scope": rows[0][0]["scope"]}, prepare=prepare
                )
            except MalformedModelResponse:
                self.reset_budget_backoff()
                if len(sent) <= 1:
                    raise
                raw = None
            else:
                self.reset_budget_backoff()
            if len(sent) > 1:
                verdicts = batch_verdicts(raw, {item.proposal["id"] for item in sent})
            else:
                verdicts = {sent[0].proposal["id"]: review_result(raw)}

            for item in sent:
                pid = item.proposal["id"]
                if pid not in verdicts:
                    await self.engine.review_write(lambda: self.queue.fallback(pid, lease))
                    continue
                try:
                    await self.engine.commit_review(
                        item,
                        verdicts[pid],
                        automatic=True,
                        valid=lambda: (
                            self.enabled and self.queue.owns(pid, lease)
                            and ("review", pid) not in self.engine.operation_locks
                        ),
                        receipt=lambda p, _: self.queue.complete(p["id"], lease),
                    )
                except (Conflict, ReviewUnavailable):
                    # Stale or manually claimed results cannot block matching sibling verdicts.
                    continue
                except Exception as exc:
                    self.engine.last_error = type(exc).__name__
                    retry = time.time() + self.engine.review_retry_interval()
        except ReviewDeferred as exc:
            retry = self.note_budget_busy(exc)
        except ReviewUnavailable:
            pass
        except Exception as exc:
            if series_panels.is_provider_budget_busy_error(exc):
                retry = self.note_budget_busy(exc)
            else:
                self.engine.last_error = type(exc).__name__
                retry = time.time() + self.engine.review_retry_interval()
        finally:
            await self.engine.review_write(lambda: self.queue.release(lease, retry))

    def finished(self, task):
        self.active.pop(task, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                self.engine.last_error = type(error).__name__
        self.wake.set()

    async def run(self):
        self.running = True
        next_reconcile = 0
        try:
            while not self.engine.closed:
                self.wake.clear()
                interval = self.engine.review_retry_interval()
                delay = interval if self.enabled else 5
                slots = 2 - len(self.active)
                backoff_rem = self.budget_backoff_remaining() if self.enabled else 0.0
                if backoff_rem > 0.0:
                    delay = backoff_rem
                elif self.enabled and slots:
                    busy = {pid for pids in self.active.values() for pid in pids}
                    now = time.time()
                    discover = self.reconcile or now >= next_reconcile
                    if discover:
                        self.reconcile = False
                        next_reconcile = now + interval
                    try:
                        groups, next_at = await self.engine.review_write(
                            lambda: self.ready(now, busy, discover=discover)
                        )
                    except Exception as exc:
                        self.engine.last_error = type(exc).__name__
                        self.reconcile |= discover
                        delay = min(5, interval)
                    else:
                        delay = max(0, min(next_reconcile, next_at) - time.time())
                        for rows in groups[:slots]:
                            task = asyncio.create_task(self.dispatch(rows))
                            self.active[task] = {row["proposal"] for row, _ in rows}
                            task.add_done_callback(self.finished)
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=delay)
                except TimeoutError:
                    continue
        finally:
            self.running = False
            tasks = list(self.active)
            for task in tasks:
                task.cancel()
            await self.engine.finish_operation(asyncio.gather(*tasks, return_exceptions=True))
