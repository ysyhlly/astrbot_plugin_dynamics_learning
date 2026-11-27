"""Dynamic subscription bridge for Chat Dynamics Event Bus (zero hard coupling)."""

from __future__ import annotations

import inspect
import time
from collections import deque
from typing import Any

from .contract import candidate, policy_allowed
from .. import series_panels

EVENT_BUS_DAILY_PROPOSAL_CAP = 6
VIBE_TTL_SECONDS = 1800.0
GAME_TTL_SECONDS = 7200.0
SUBSCRIBED_EVENTS = ("topic_archived", "game_state_changed", "vibe_transition")
_COMPLETED_GAME_PHASES = frozenset({"completed", "finished", "settled", "ended", "milestone", "closed"})


class DynamicsEventBridge:
    """Subscribes to Chat Dynamics Event Bus and converts signals into reviewed learning clues."""

    def __init__(self, engine: Any, context: Any, *, owner_id: str = "astrbot_plugin_dynamics_learning"):
        self.engine = engine
        self.context = context
        self.owner_id = owner_id
        self._tokens: list[tuple[Any, str, str]] = []  # (chat_owner, event_type, token)
        self._subscribed_owner_id: int | None = None
        self.received_counts: dict[str, int] = {e: 0 for e in SUBSCRIBED_EVENTS}
        self.proposals_created: int = 0
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=30)

    def attach(self) -> list[str]:
        """Idempotently subscribe to topic_archived, game_state_changed, and vibe_transition."""
        if self.context is None or self.engine is None or self.engine.closed:
            return []
        owner = series_panels.find_plugin_owner(self.context, series_panels.CHAT_DYNAMICS_ID)
        if owner is None:
            return []
        if self._subscribed_owner_id == id(owner) and self._tokens:
            return [ev for _, ev, _ in self._tokens]
        if self._tokens:
            self.detach()
        subscribe = getattr(owner, "subscribe_dynamics_event", None)
        if not callable(subscribe) or inspect.iscoroutinefunction(subscribe):
            return []

        handlers = {
            "topic_archived": self.on_topic_archived,
            "game_state_changed": self.on_game_state_changed,
            "vibe_transition": self.on_vibe_transition,
        }
        attached: list[str] = []
        for event_type, handler in handlers.items():
            try:
                token = subscribe(event_type, handler, owner=self.owner_id)
            except TypeError:
                try:
                    token = subscribe(event_type, handler)
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(token, str) and token:
                self._tokens.append((owner, event_type, token))
                attached.append(event_type)
        if attached:
            self._subscribed_owner_id = id(owner)
        return attached

    def detach(self) -> int:
        """Safely unsubscribe all active event-bus tokens on plugin unload."""
        removed = 0
        tokens = list(self._tokens)
        self._tokens.clear()
        self._subscribed_owner_id = None
        for owner, _event_type, token in tokens:
            unsub = getattr(owner, "unsubscribe_dynamics_event", None)
            if callable(unsub) and not inspect.iscoroutinefunction(unsub):
                try:
                    if unsub(token):
                        removed += 1
                except Exception:
                    continue
        return removed

    def status(self) -> dict[str, Any]:
        return {
            "connected": bool(self._tokens),
            "subscribed_events": [ev for _, ev, _ in self._tokens],
            "received_counts": dict(self.received_counts),
            "proposals_created": self.proposals_created,
            "recent_count": len(self.recent_events),
        }

    def _scope_eligible(self, umo: str) -> bool:
        clean = str(umo or "").strip()
        if not clean or ":GroupMessage:" not in clean or self.engine is None or self.engine.closed:
            return False
        if series_panels.is_series_group_excluded(self.context, umo=clean):
            return False
        return bool(self.engine.allowed(clean))

    def _filter_participants(self, umo: str, raw_participants: Any, *, limit: int = 4) -> list[str]:
        if not isinstance(raw_participants, (list, tuple, set)):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw_participants:
            if isinstance(item, dict):
                mid = str(item.get("user_id") or item.get("member_id") or item.get("id") or "").strip()
            else:
                mid = str(item or "").strip()
            if not mid or len(mid) > 128 or mid in seen:
                continue
            seen.add(mid)
            if self.engine.is_privacy_muted(umo, mid):
                continue
            out.append(mid)
            if len(out) >= limit:
                break
        return out

    def _propose_relationship_clue(
        self,
        *,
        scope: str,
        persona: str,
        member: str,
        key: str,
        value: str,
        reason: str,
        evidence_id: str,
        evidence_text: str,
        source_kind: str,
    ) -> dict[str, Any] | None:
        ev = {
            "id": evidence_id,
            "scope": scope,
            "persona": persona,
            "member": member,
            "message_id": evidence_id,
            "turn_id": evidence_id,
            "kind": source_kind,
            "baseline": "event_bus",
            "host_version": "event_bus",
            "baseline_threshold": 0,
            "text": evidence_text[:1000],
            "timestamp": time.time(),
            "context": {
                "source_verified": False,
                "character_scope": dict(self.engine.character_scopes.get((scope, persona)) or {}),
            },
        }
        raw = {
            "category": "relationship",
            "key": key[:80],
            "member": member,
            "value": value[:500],
            "reason": reason[:500],
            "days": 30,
            "evidence_ids": [evidence_id],
        }
        data = candidate(raw, [ev], None)
        if not policy_allowed(data):
            return None
        data["source_kind"] = source_kind
        mode = "human" if self.engine.mode("relationship") == "human" else "llm_then_human"
        prop = self.engine.store.propose(data, mode, f"system:{source_kind}")
        self.proposals_created += 1
        return prop

    def on_vibe_transition(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.received_counts["vibe_transition"] += 1
        if not isinstance(payload, dict) or self.engine is None or self.engine.closed:
            return {"ok": False, "reason": "invalid_payload"}
        umo = str(payload.get("umo") or "").strip()[:512]
        if not umo:
            return {"ok": False, "reason": "missing_umo"}
        now = time.time()
        entry = {
            "old_vibe": str(payload.get("old_vibe") or "").strip()[:64],
            "new_vibe": str(payload.get("new_vibe") or "").strip()[:64],
            "serious_mode": bool(payload.get("serious_mode", False)),
            "updated_at": now,
            "expires_at": now + VIBE_TTL_SECONDS,
        }
        self.engine.scope_vibes[umo] = entry
        self.recent_events.append({"event": "vibe_transition", "umo": umo, "vibe": entry["new_vibe"], "at": now})
        return {"ok": True, "umo": umo, **entry}

    def on_topic_archived(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.received_counts["topic_archived"] += 1
        if not isinstance(payload, dict) or self.engine is None or self.engine.closed:
            return {"ok": False, "proposals": 0, "reason": "invalid_payload"}
        umo = str(payload.get("umo") or "").strip()[:512]
        if not self._scope_eligible(umo):
            return {"ok": False, "proposals": 0, "reason": "scope_not_enabled"}
        summary = " ".join(str(payload.get("summary") or "").split())[:300]
        topic_id = str(payload.get("topic_id") or "").strip()[:64] or str(int(time.time() * 1000))
        try:
            msg_count = int(payload.get("message_count") or 0)
        except (TypeError, ValueError):
            msg_count = 0
        if not summary or (msg_count and msg_count < 2):
            return {"ok": True, "proposals": 0, "reason": "low_signal"}

        persona = self.engine.current_personas.get(umo) or self.engine.shared_persona_id()
        if not persona:
            return {"ok": False, "proposals": 0, "reason": "persona_unavailable"}

        participants = self._filter_participants(umo, payload.get("participants"), limit=4)
        if not participants:
            return {"ok": True, "proposals": 0, "reason": "no_eligible_participants"}

        used_today = self.engine.store.count_event_bus_proposals_today(umo)
        remaining = max(0, EVENT_BUS_DAILY_PROPOSAL_CAP - used_today)
        if remaining <= 0:
            return {"ok": True, "proposals": 0, "reason": "daily_cap_reached"}

        topic_label = summary[:24]
        created_ids: list[str] = []
        for member in participants[:remaining]:
            try:
                prop = self._propose_relationship_clue(
                    scope=umo,
                    persona=persona,
                    member=member,
                    key=f"同好话题:{topic_label}",
                    value=f"在归档群聊话题中与群友积极交流：{summary[:180]}",
                    reason=f"来自核心事件总线 topic_archived（#{topic_id}，{msg_count or '多'} 条消息），严格遵循固定性格红线并进入审查流。",
                    evidence_id=f"eb:topic:{topic_id}:{member}",
                    evidence_text=f"[归档话题 {topic_id}] {summary}",
                    source_kind="event_bus:topic_archived",
                )
                if prop is not None:
                    created_ids.append(prop["id"])
            except Exception:
                continue

        self.recent_events.append(
            {"event": "topic_archived", "umo": umo, "topic_id": topic_id, "proposals": len(created_ids), "at": time.time()}
        )
        return {"ok": True, "proposals": len(created_ids), "proposal_ids": created_ids}

    def on_game_state_changed(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.received_counts["game_state_changed"] += 1
        if not isinstance(payload, dict) or self.engine is None or self.engine.closed:
            return {"ok": False, "proposals": 0, "reason": "invalid_payload"}
        umo = str(payload.get("umo") or "").strip()[:512]
        if not umo:
            return {"ok": False, "proposals": 0, "reason": "missing_umo"}

        now = time.time()
        active = bool(payload.get("active", False))
        engine_name = str(payload.get("engine") or "group_game").strip()[:48]
        activity_id = str(payload.get("activity_id") or "").strip()[:64] or str(int(now * 1000))
        title = " ".join(str(payload.get("title") or "").split())[:80]
        phase = str(payload.get("phase") or "").strip().lower()[:32]

        self.engine.active_games[umo] = {
            "active": active,
            "engine": engine_name,
            "activity_id": activity_id,
            "title": title,
            "phase": phase,
            "updated_at": now,
            "expires_at": now + GAME_TTL_SECONDS,
        }
        self.recent_events.append(
            {"event": "game_state_changed", "umo": umo, "engine": engine_name, "phase": phase, "active": active, "at": now}
        )

        # Only turn completed / milestone group game events into reviewed relationship clues
        if not self._scope_eligible(umo):
            return {"ok": True, "proposals": 0, "reason": "scope_not_enabled"}
        if active and phase not in _COMPLETED_GAME_PHASES:
            return {"ok": True, "proposals": 0, "reason": "in_progress"}
        if not title and not engine_name:
            return {"ok": True, "proposals": 0, "reason": "empty_title"}

        persona = self.engine.current_personas.get(umo) or self.engine.shared_persona_id()
        if not persona:
            return {"ok": True, "proposals": 0, "reason": "persona_unavailable"}

        participants = self._filter_participants(umo, payload.get("participants"), limit=4)
        if not participants:
            return {"ok": True, "proposals": 0, "reason": "no_eligible_participants"}

        used_today = self.engine.store.count_event_bus_proposals_today(umo)
        remaining = max(0, EVENT_BUS_DAILY_PROPOSAL_CAP - used_today)
        if remaining <= 0:
            return {"ok": True, "proposals": 0, "reason": "daily_cap_reached"}

        label = title or engine_name
        created_ids: list[str] = []
        for member in participants[:remaining]:
            try:
                prop = self._propose_relationship_clue(
                    scope=umo,
                    persona=persona,
                    member=member,
                    key=f"共玩同好:{label[:24]}",
                    value=f"参与了群内「{label}」（{engine_name}）共玩/协作活动，展现出该主题互动兴趣。",
                    reason=f"来自核心事件总线 game_state_changed（{engine_name}/{phase or 'ended'}），进入现有审查流。",
                    evidence_id=f"eb:game:{activity_id}:{phase or 'ended'}:{member}",
                    evidence_text=f"[群活动 {engine_name}] {label} ({phase or 'ended'})",
                    source_kind="event_bus:game_state_changed",
                )
                if prop is not None:
                    created_ids.append(prop["id"])
            except Exception:
                continue
        return {"ok": True, "proposals": len(created_ids), "proposal_ids": created_ids}
