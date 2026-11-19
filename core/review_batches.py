"""Durable review jobs with cooperative controls and explicit failed-only retries."""

import asyncio
import json
import time
import uuid

from .store import Conflict

ACTIVE = {"queued", "running", "pausing", "paused", "cancelling"}
UNFINISHED = {"queued", "running", "interrupted"}
RESULTS = {"reviewed", "applied", "rejected", "deferred"}


class ReviewUnavailable(ValueError):
    pass


class ReviewBatches:
    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store
        with self.store.lock:
            old = [
                json.loads(r[0])
                for r in self.store.db.execute(
                    "SELECT body FROM review_batches WHERE status IN ('queued','running','pausing','cancelling')"
                )
            ]
        for job in old:
            self.interrupt(job)
        # Adopt unfinished jobs on upgrade; newer retry batches own the reservation.
        with self.store.transaction():
            seeded = self.store.db.execute("SELECT 1 FROM analysis_meta WHERE key='review_claims_v1'").fetchone()
            for row in (
                [] if seeded else self.store.db.execute("SELECT body FROM review_batches ORDER BY created").fetchall()
            ):
                job = json.loads(row[0])
                for item in job["items"]:
                    if item["status"] in UNFINISHED | {"failed", "cancelled"}:
                        p = self.store.db.execute("SELECT state FROM proposals WHERE id=?", (item["id"],)).fetchone()
                        if p and p[0] in {"pending", "reviewed"}:
                            self.store.db.execute(
                                "INSERT OR REPLACE INTO review_claims VALUES(?,?,?)",
                                (item["id"], job["scope"], job["id"]),
                            )
            self.store.db.execute("INSERT OR IGNORE INTO analysis_meta VALUES('review_claims_v1','1')")

    def get(self, identity):
        with self.store.lock:
            row = self.store.db.execute("SELECT body FROM review_batches WHERE id=?", (identity,)).fetchone()
            return json.loads(row[0]) if row else None

    def save(self, job):
        job["updated"] = time.time()
        with self.store.transaction():
            return (
                self.store.db.execute(
                    "UPDATE review_batches SET status=?,body=? WHERE id=?",
                    (job["status"], json.dumps(job, ensure_ascii=False), job["id"]),
                ).rowcount
                == 1
            )

    def finish_cancel(self, job):
        job["status"] = "cancelled"
        for item in job["items"]:
            if item["status"] in UNFINISHED:
                item["status"] = "cancelled"
        self.save(job)

    def interrupt(self, job):
        current = self.get(job["id"])
        if current is None:
            return
        if current["status"] == "cancelling":
            self.finish_cancel(current)
        elif current["status"] not in {"paused", "cancelled", "completed", "completed_with_issues"}:
            if not any(i["status"] in UNFINISHED for i in current["items"]):
                current["status"] = self.completed_status(current)
            else:
                current["status"] = "paused" if current["status"] == "pausing" else "interrupted"
                for item in current["items"]:
                    if item["status"] in {"queued", "running"}:
                        item["status"] = "interrupted"
            self.save(current)
        job.update(current)

    def ensure_available(self, scope, exclude=""):
        with self.store.lock:
            if self.store.db.execute(
                "SELECT 1 FROM review_batches WHERE scope=? AND id!=? AND status IN ('queued','running','pausing','paused','cancelling')",
                (scope, exclude),
            ).fetchone():
                raise ValueError("本群已有批量审查任务，请先完成或取消该任务")

    def create(self, scope, proposals, actor, parent=None, apply_only=()):
        with self.store.transaction():
            self.ensure_available(scope)
            job = {
                "id": uuid.uuid4().hex,
                "scope": scope,
                "actor": actor,
                "created": time.time(),
                "status": "queued",
                "controls": [],
                "items": [
                    {
                        "id": p["id"],
                        "updated": p["updated"],
                        "member": p["member"],
                        "key": p["key"],
                        "status": "queued",
                        "error": "",
                    }
                    for p in proposals
                ],
            }
            for item in job["items"]:
                if item["id"] in apply_only:
                    item["apply_only"] = True
            if parent:
                job["parent_id"] = parent["id"]
            self.store.db.execute(
                "INSERT INTO review_batches VALUES(?,?,?,?,?)",
                (job["id"], scope, job["status"], job["created"], json.dumps(job, ensure_ascii=False)),
            )
            for p in proposals:
                self.store.db.execute("INSERT OR REPLACE INTO review_claims VALUES(?,?,?)", (p["id"], scope, job["id"]))
            if parent:
                parent["retries"] = (parent.get("retries", []) + [job["id"]])[-100:]
                self.save(parent)
            return job

    async def prepare(self, scope, items, actor):
        async with self.engine.lock:
            self.ensure_available(scope)
            proposals = await self.engine.batch_snapshot(scope, items, actor)
            if any(("review", p["id"]) in self.engine.operation_locks for p in proposals):
                raise ValueError("所选条目已有审查正在进行，请稍后刷新或取消勾选该条目")
            if any(not self.engine.can_llm_review(p) for p in proposals):
                raise ValueError("所选记录不需要或暂不能进行 LLM 审查，请刷新选择")
            return self.create(scope, proposals, actor)

    async def control(self, identity, action, actor, *, require_provider=False):
        async with self.engine.lock:
            job = self.get(identity)
            if not actor.startswith("human:") or not job or not self.engine.allowed(job["scope"]):
                raise ValueError("任务不存在或群已停用")
            status = job["status"]
            if action == "pause":
                if status not in {"queued", "running", "pausing", "paused"}:
                    raise ValueError("任务当前不可暂停")
                job["status"] = "paused" if status in {"queued", "paused"} else "pausing"
            elif action == "cancel":
                if status not in ACTIVE | {"interrupted", "cancelled"}:
                    raise ValueError("已结束的任务无需取消")
                if status in {"running", "pausing", "cancelling"}:
                    job["status"] = "cancelling"
                else:
                    self.finish_cancel(job)
            elif action == "resume":
                if status not in {"paused", "interrupted"}:
                    raise ValueError("请等待当前项结束并暂停后再恢复")
                self.ensure_available(job["scope"], job["id"])
                for item in job["items"]:
                    if item["status"] == "interrupted":
                        item["status"] = "queued"
                if not any(i["status"] == "queued" for i in job["items"]):
                    raise ValueError("没有未完成项；失败项请使用一键重试")
                if (
                    require_provider
                    and not self.engine.provider_for("review_provider")
                    and any(i["status"] == "queued" and not i.get("apply_only") for i in job["items"])
                ):
                    raise ValueError("请先配置审查模型")
                job["status"] = "queued"
            elif action == "retry":
                if status in ACTIVE:
                    raise ValueError("请先结束或取消当前任务，再重试失败项")
                if job.get("retries"):
                    raise ValueError("已有重试批次，请在最新重试批次继续操作")
                rows = []
                excluded = []
                apply_ids = set()
                for item in job["items"]:
                    apply_pending = item["status"] == "reviewed" and item.get("auto_apply_pending")
                    if item["status"] != "failed" and not apply_pending:
                        continue
                    apply_only = apply_pending or item.get("apply_only")
                    stamp = item.get("result_updated") if apply_pending else item["updated"]
                    try:
                        p = await asyncio.to_thread(self.store.get, item["id"])
                    except ValueError:
                        excluded.append(item["id"])
                        continue
                    eligible = self.engine.can_retry_apply if apply_only else self.engine.can_llm_review
                    if p["updated"] != stamp or not eligible(p, owner=job["id"]):
                        excluded.append(item["id"])
                        continue
                    rows.append(p)
                    if apply_only:
                        apply_ids.add(p["id"])
                if not rows:
                    raise ValueError("没有可重试的记录；提案可能已被处理或修改")
                if require_provider and not self.engine.provider_for("review_provider") and len(apply_ids) != len(rows):
                    raise ValueError("请先配置审查模型")
                with self.store.transaction():
                    child = self.create(job["scope"], rows, actor, parent=job, apply_only=apply_ids)
                    child["retry_skipped"] = excluded
                    self.save(child)
                return child
            else:
                raise ValueError("invalid job action")
            job["controls"] = (job.get("controls", []) + [{"action": action, "actor": actor, "at": time.time()}])[-100:]
            self.save(job)
            return job

    def recent(self, scope=""):
        scopes = [s for s in self.engine.effective_sessions() if self.engine.allowed(s)]
        if scope and scope not in scopes:
            raise ValueError("scope not enabled")
        with self.store.lock:
            # Paused/active jobs remain discoverable even after many completed jobs elsewhere.
            return [
                json.loads(r[0])
                for r in self.store.db.execute(
                    "SELECT body FROM review_batches WHERE scope IN (SELECT value FROM json_each(?)) AND (?='' OR scope=?) ORDER BY CASE WHEN status IN ('queued','running','pausing','paused','cancelling') THEN 0 ELSE 1 END,created DESC LIMIT 10",
                    (json.dumps(scopes), scope, scope),
                )
            ]

    def permits_result(self, identity):
        job = self.get(identity)
        return bool(job and job["status"] in {"running", "pausing"})

    def started(self, identity, proposal):
        job = self.get(identity)
        if job:
            item = next(i for i in job["items"] if i["id"] == proposal["id"])
            # Profile mode adoption changes the stamp before the call. Retry this exact version.
            item["updated"] = proposal["updated"]
            if not self.save(job):
                raise ReviewUnavailable("batch removed during mode adoption")
        else:
            raise ReviewUnavailable("batch removed during mode adoption")

    def committed(self, identity, proposal, result):
        """Called inside the SAME transaction that records/applies the LLM verdict."""
        job = self.get(identity)
        if not job or job["status"] not in {"running", "pausing"}:
            raise ReviewUnavailable("batch no longer accepts results")
        item = next(i for i in job["items"] if i["id"] == proposal["id"])
        item.update(
            status="applied"
            if proposal["state"] == "applied"
            else {"approve": "reviewed", "reject": "rejected", "defer": "deferred"}[result["verdict"]],
            error="",
            committed=True,
            result_updated=proposal["updated"],
            verdict=result["verdict"],
            auto_apply_pending=proposal["state"] == "reviewed" and proposal["mode"] == "llm",
        )
        if not self.save(job):
            raise ReviewUnavailable("batch removed during result commit")
        # Keep reviewed work reserved until the batch records automatic application.
        # Maintenance must not apply it outside the transaction that records its receipt.
        if not item["auto_apply_pending"]:
            self.store.db.execute("DELETE FROM review_claims WHERE proposal=? AND job=?", (proposal["id"], identity))

    @staticmethod
    def completed_status(job):
        return (
            "completed_with_issues"
            if any(
                i["status"] in {"failed", "skipped"} or i.get("error") or i.get("auto_apply_pending")
                for i in job["items"]
            )
            else "completed"
        )

    def _select_batch_chunk(self, queued, single_only_ids):
        first = queued[0]
        if first.get("apply_only") or first["id"] in single_only_ids:
            return [first]
        try:
            first_p = self.store.get(first["id"])
        except ValueError:
            return [first]
        target_pair = (first_p["scope"], first_p["persona"])
        from .engine import REVIEW_BATCH_MAX

        candidates = [(first, first_p)]
        for item in queued[1:]:
            if len(candidates) >= REVIEW_BATCH_MAX:
                break
            if item.get("apply_only") or item["id"] in single_only_ids:
                break
            try:
                p = self.store.get(item["id"])
            except ValueError:
                break
            if (p["scope"], p["persona"]) != target_pair:
                break
            candidates.append((item, p))
        if len(candidates) < 2:
            return [first]
        chunks = self.engine._pack_review_chunks(target_pair[0], target_pair[1], [p for _, p in candidates])
        if not chunks or len(chunks[0]) < 2:
            return [first]
        chosen_ids = {p["id"] for p in chunks[0]}
        return [item for item, p in candidates if p["id"] in chosen_ids]

    async def _run_batch_chunk(self, identity, job, current, batch_items, single_only_ids):
        current["status"] = "running"
        batch_ids = {i["id"] for i in batch_items}
        for item in current["items"]:
            if item["id"] in batch_ids:
                item["status"] = "running"
        if not self.save(current):
            return
        chunk_specs = [{"id": i["id"], "expected_updated": i["updated"]} for i in batch_items]
        try:
            outcome = await self.engine._execute_review_chunk(
                chunk_specs,
                owner=identity,
                valid=lambda: self.permits_result(identity),
                started=lambda p: self.started(identity, p),
                receipt=lambda p, r: self.committed(identity, p, r),
                require_llm=True,
            )
        except ReviewUnavailable:
            outcome = {
                "committed": {},
                "unavailable": {i["id"]: ReviewUnavailable("batch unavailable") for i in batch_items},
                "fallback": [],
                "errors": {},
            }
        except Exception as exc:
            outcome = {
                "committed": {},
                "unavailable": {},
                "fallback": [],
                "errors": {i["id"]: exc for i in batch_items},
            }
        current = self.get(identity)
        if not current:
            return
        cancelling = current["status"] == "cancelling"
        for batch_item in batch_items:
            pid = batch_item["id"]
            item = next(i for i in current["items"] if i["id"] == pid)
            if item.get("committed") and item["status"] in RESULTS:
                if pid in outcome["errors"]:
                    err = type(outcome["errors"][pid]).__name__
                    item["error"] = "审查结果已保存；后续生效或刷新未完成（" + err + "）"
                continue
            if pid in outcome["fallback"]:
                if cancelling:
                    item.update(status="cancelled", error="已取消，未提交审查结果")
                else:
                    item.update(status="queued", error="")
                    single_only_ids.add(pid)
            elif pid in outcome["unavailable"]:
                if cancelling:
                    item.update(status="cancelled", error="已取消，未提交审查结果")
                else:
                    item.update(status="skipped", error="记录、审查设置或版本已变化；请刷新核对")
            elif pid in outcome["errors"]:
                if cancelling:
                    item.update(status="cancelled", error="已取消，未提交审查结果")
                else:
                    item.update(status="failed", error=type(outcome["errors"][pid]).__name__)
            elif cancelling:
                item.update(status="cancelled", error="已取消，未提交审查结果")
            else:
                item.update(status="queued", error="")
                single_only_ids.add(pid)
        self.save(current)
        job.update(current)

    async def run(self, job):
        identity = job["id"]
        single_only_ids = set()
        async with self.engine.operation("review-batch", identity):
            try:
                while True:
                    current = self.get(identity)
                    if not current:
                        return
                    job.update(current)
                    status = current["status"]
                    if status == "cancelling":
                        self.finish_cancel(current)
                        job.update(current)
                        return
                    if status not in {"queued", "running", "pausing"}:
                        return
                    if self.engine.closed:
                        self.interrupt(job)
                        return
                    if any(i["status"] == "running" for i in current["items"]):
                        # A previous worker vanished; do not silently mark or skip its work.
                        self.interrupt(job)
                        return
                    queued = [i for i in current["items"] if i["status"] == "queued"]
                    if not queued:
                        current["status"] = self.completed_status(current)
                        self.save(current)
                        job.update(current)
                        return
                    if status == "pausing":
                        current["status"] = "paused"
                        self.save(current)
                        job.update(current)
                        return
                    batch_items = self._select_batch_chunk(queued, single_only_ids)
                    if len(batch_items) >= 2:
                        await self._run_batch_chunk(identity, job, current, batch_items, single_only_ids)
                        continue
                    pending = queued[0]
                    current["status"] = "running"
                    pending["status"] = "running"
                    if not self.save(current):
                        return
                    error = ""
                    try:
                        apply_only = bool(pending.get("apply_only"))
                        p = await self.engine.review(
                            pending["id"],
                            expected_updated=pending["updated"],
                            require_llm=not apply_only,
                            apply_only=apply_only,
                            owner=identity,
                            valid=lambda: self.permits_result(identity),
                            started=lambda p: self.started(identity, p),
                            receipt=lambda p, r: self.committed(identity, p, r),
                        )
                        verdict = next(
                            (r["verdict"] for r in reversed(p["reviews"]) if r["actor"].startswith("llm:")), "defer"
                        )
                        outcome = (
                            "applied"
                            if p["state"] == "applied"
                            else {"approve": "reviewed", "reject": "rejected", "defer": "deferred"}[verdict]
                        )
                    except (ReviewUnavailable, Conflict, KeyError):
                        outcome, error = "skipped", "记录、审查设置或版本已变化；请刷新核对"
                    except Exception as exc:
                        outcome, error = "failed", type(exc).__name__
                    # Read controls again after the await; never overwrite a pause/cancel with stale state.
                    current = self.get(identity)
                    if not current:
                        return
                    item = next(i for i in current["items"] if i["id"] == pending["id"])
                    if item.get("committed") and item["status"] in RESULTS:
                        # Refresh/apply failures or cancellation after commit cannot erase its receipt.
                        if error:
                            item["error"] = "审查结果已保存；后续生效或刷新未完成（" + error + "）"
                    elif current["status"] == "cancelling" and outcome in {"failed", "skipped"}:
                        outcome, error = "cancelled", "已取消，未提交审查结果"
                        item.update(status=outcome, error=error)
                    else:
                        item.update(status=outcome, error=error)
                    self.save(current)
                    job.update(current)
            except asyncio.CancelledError:
                self.interrupt(job)
                raise
            except Exception:
                self.interrupt(job)
                raise
