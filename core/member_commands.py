"""Member-facing C-end command helpers: /我的印象, /群黑话, /纠正黑话, /习惯纠偏, /学习静音."""

from __future__ import annotations

import inspect
import re
import time
from pathlib import Path
from typing import Any

from .contract import candidate, is_pure_ascii, jargon_matches, policy_allowed
from .impression_card import format_impression_text, render_impression_card
from .series_bridge import resolve_persona_labels
from .. import series_panels

DEFAULT_MEMBER_CORRECTION_DAILY_LIMIT = 10


def _member_correction_limit(config: dict[str, Any]) -> int:
    raw = config.get("member_correction_daily_limit", DEFAULT_MEMBER_CORRECTION_DAILY_LIMIT)
    if isinstance(raw, bool):
        return DEFAULT_MEMBER_CORRECTION_DAILY_LIMIT
    try:
        return max(1, min(50, int(raw)))
    except (TypeError, ValueError):
        return DEFAULT_MEMBER_CORRECTION_DAILY_LIMIT


def _parse_jargon_correction_args(raw: str) -> tuple[str, str]:
    clean = str(raw or "").strip()
    if not clean:
        return "", ""
    # Support "词 释义" or "词：释义" or "词: 释义"
    sep_match = re.match(r"^([^\s:：]+)\s*[:：]\s*(.+)$", clean, re.DOTALL)
    if sep_match:
        return sep_match.group(1).strip(), sep_match.group(2).strip()
    parts = clean.split(None, 1)
    if len(parts) < 2:
        return parts[0].strip(), ""
    return parts[0].strip(), parts[1].strip()


def build_my_impression_payload(
    engine: Any,
    context: Any,
    *,
    scope: str,
    member: str,
    member_name: str = "",
) -> dict[str, Any]:
    """Assemble strictly self-only, current-group, approved & sanitized impression data."""
    if engine is None or engine.closed:
        return {"ok": False, "reason": "群间学习尚未就绪，请稍后再试。"}
    if ":GroupMessage:" not in str(scope or "") or not engine.allowed(scope):
        return {"ok": False, "reason": "当前群未启用群间学习，请在已启用的群聊中使用。"}
    if engine.is_privacy_muted(scope, member):
        return {
            "ok": False,
            "muted": True,
            "reason": "你已在本群开启学习静音，已暂停展示与采集你的个人学习资料。发送「/学习静音 关闭」可恢复。",
        }
    persona = engine.current_personas.get(scope) or engine.shared_persona_id()
    if not persona:
        return {"ok": False, "reason": "当前群尚未取得角色快照，请先在群内正常聊天后再查看。"}

    now = time.time()
    rows = [
        e
        for e in engine._entries_for(scope, persona)
        if e.get("member") == member
        and e.get("active")
        and policy_allowed(e)
        and (e.get("category") == "affection" or float(e.get("expires") or 0) > now)
    ]
    rows.sort(key=lambda e: float(e.get("state_updated_at") or e.get("revision") or 0), reverse=True)

    likes: list[str] = []
    dislikes: list[str] = []
    boundaries: list[str] = []
    habits: list[str] = []
    seen_topics: dict[str, set[str]] = {"like": set(), "dislike": set(), "boundary": set()}

    for entry in rows:
        cat = entry.get("category")
        val = entry.get("value")
        if cat == "preference" and isinstance(val, dict):
            stance = str(val.get("stance") or "").strip().lower()
            topic = " ".join(str(val.get("topic") or "").split())[:48]
            cond = " ".join(str(val.get("condition") or "").split())[:32]
            if not topic or stance not in seen_topics:
                continue
            label = f"{topic}（{cond}）" if cond else topic
            if topic.casefold() in seen_topics[stance]:
                continue
            seen_topics[stance].add(topic.casefold())
            if stance == "like":
                likes.append(label)
            elif stance == "dislike":
                dislikes.append(label)
            elif stance == "boundary":
                boundaries.append(label)
        elif cat == "typing_habit":
            if isinstance(val, dict) and val.get("kind") == "observed_typing" and val.get("sample_count"):
                short_pct = round(100 * float(val.get("short_rate") or 0))
                emoji_pct = round(100 * float(val.get("emoji_rate") or 0))
                mean_chars = val.get("mean_chars", 0)
                habits.append(
                    f"平均 {mean_chars} 字/条 · 短句占比 {short_pct}% · 表情占比 {emoji_pct}%（样本 {val['sample_count']} 条）"
                )
            elif isinstance(val, str) and val.strip():
                habits.append(" ".join(val.split())[:80])
        elif cat == "style" and isinstance(val, str) and val.strip():
            habits.append(" ".join(val.split())[:80])

    aff_info: dict[str, Any] | None = None
    if engine.social is not None:
        aff_entry = engine.social.current_affection(scope, persona, member)
        if aff_entry and isinstance(aff_entry.get("value"), dict):
            aff_val = aff_entry["value"]
            aff_info = {
                "level": float(aff_val.get("affection_level") or 0.0),
                "tone": float(aff_val.get("tone") or 0.0)
                if time.time() - float(aff_entry.get("tone_updated_at") or 0) < 86400
                else None,
            }

    labels = resolve_persona_labels(context, engine.store, engine)
    persona_name = labels.get(persona) or (persona[:12] if len(persona) > 16 else persona)
    resolved_member_name = str(member_name or "").strip()
    if not resolved_member_name and engine.social is not None:
        resolved_member_name = str(
            engine.social.groups.get(scope, {}).get("users", {}).get(member, {}).get("name") or ""
        ).strip()
    if not resolved_member_name:
        resolved_member_name = member

    payload = {
        "ok": True,
        "scope": scope,
        "member": member,
        "member_name": resolved_member_name,
        "persona": persona,
        "persona_name": persona_name,
        "affection": aff_info,
        "likes": likes[:8],
        "dislikes": dislikes[:6],
        "boundaries": boundaries[:6],
        "habits": habits[:4],
    }
    payload["text"] = format_impression_text(payload)
    card_dir = Path(engine.store.path).parent / "cards"
    card_path = render_impression_card(payload, card_dir)
    payload["image_path"] = str(card_path) if card_path is not None else ""
    return payload


def query_group_jargon_list(engine: Any, *, scope: str, keyword: str = "", limit: int = 15) -> dict[str, Any]:
    if engine is None or engine.closed:
        return {"ok": False, "text": "群间学习尚未就绪，请稍后再试。", "terms": []}
    if ":GroupMessage:" not in str(scope or "") or not engine.allowed(scope):
        return {"ok": False, "text": "当前群未启用群间学习。", "terms": []}
    if not engine.config.get("jargon_enabled", True):
        return {"ok": False, "text": "本群黑话词库功能已关闭。", "terms": []}
    persona = engine.current_personas.get(scope) or engine.shared_persona_id()
    if not persona:
        return {"ok": False, "text": "当前群尚未取得角色快照，暂无已确认黑话。", "terms": []}

    clean_kw = str(keyword or "").strip()[:64]
    now = time.time()
    active_jargon = [
        e
        for e in engine._entries_for(scope, persona)
        if e.get("category") == "jargon" and e.get("active") and float(e.get("expires") or 0) > now
    ]
    active_jargon.sort(key=lambda e: float(e.get("state_updated_at") or e.get("revision") or 0), reverse=True)

    if clean_kw:
        kw_cf = clean_kw.casefold()
        matched = [
            e
            for e in active_jargon
            if kw_cf in str(e.get("key") or "").casefold()
            or kw_cf in str(e.get("value") or "").casefold()
            or jargon_matches(str(e.get("key") or ""), clean_kw)
        ][: max(1, min(15, int(limit)))]
    else:
        matched = active_jargon[: max(1, min(15, int(limit)))]

    terms = [{"term": e["key"], "meaning": str(e["value"]), "revision": e["revision"]} for e in matched]
    if not terms:
        msg = (
            f"未找到与「{clean_kw}」相关的已批准群黑话。发送「/纠正黑话 <词> <释义>」可提交补充或纠偏。"
            if clean_kw
            else "本群暂无已批准的黑话词条。发送「/纠正黑话 <词> <释义>」可提交新词或释义纠偏。"
        )
        return {"ok": True, "terms": [], "text": msg}

    header = f"【本群黑话 · 匹配「{clean_kw}」({len(terms)}条)】" if clean_kw else f"【本群已批准黑话（共 {len(active_jargon)} 条，展示前 {len(terms)} 条）】"
    body = [f"• {t['term']}：{t['meaning']}" for t in terms]
    footer = "提示：发现释义不准可发「/纠正黑话 <词> <释义>」提交审查队列。"
    return {"ok": True, "terms": terms, "text": "\n".join([header, *body, footer])}


def submit_member_jargon_correction(
    engine: Any,
    *,
    scope: str,
    member: str,
    raw_args: str,
) -> dict[str, Any]:
    if engine is None or engine.closed:
        return {"ok": False, "text": "群间学习尚未就绪，请稍后再试。"}
    if ":GroupMessage:" not in str(scope or "") or not engine.allowed(scope):
        return {"ok": False, "text": "当前群未启用群间学习。"}
    if not engine.config.get("jargon_enabled", True):
        return {"ok": False, "text": "本群黑话功能已关闭。"}
    if engine.is_privacy_muted(scope, member):
        return {"ok": False, "text": "你已开启本群学习静音，请先发送「/学习静音 关闭」后再提交纠偏。"}

    term, meaning = _parse_jargon_correction_args(raw_args)
    if not term or not meaning:
        return {"ok": False, "text": "用法：/纠正黑话 <词语> <正确释义>（例如：/纠正黑话 赛博浇花 每天去3D小屋看角色浇水）"}
    if len(term) > 64 or (is_pure_ascii(term) and len(term) < 2):
        return {"ok": False, "text": "黑话词语长度需在 1-64 字内（纯英文/数字至少 2 个字符）。"}
    if len(meaning) > 500:
        return {"ok": False, "text": "释义请控制在 500 字以内。"}

    limit = _member_correction_limit(engine.config)
    actor = f"member:{member}"
    used_today = engine.store.count_member_corrections_today(scope, actor)
    if used_today >= limit:
        return {"ok": False, "text": f"今日提交纠偏已达上限（{limit} 次/天），请明天再试。"}

    persona = engine.current_personas.get(scope) or engine.shared_persona_id()
    if not persona:
        return {"ok": False, "text": "当前群尚未取得角色快照，请先在群内正常聊天后再提交。"}

    existing = next(
        (
            e
            for e in engine._entries_for(scope, persona)
            if e.get("category") == "jargon"
            and e.get("active")
            and str(e.get("key") or "").strip().casefold() == term.casefold()
        ),
        None,
    )
    key = existing["key"] if existing else term
    stamp = int(time.time() * 1000)
    ev_id = f"member_correction:jargon:{member}:{stamp}"
    event = {
        "id": ev_id,
        "scope": scope,
        "persona": persona,
        "member": "",
        "message_id": ev_id,
        "turn_id": ev_id,
        "kind": "member_correction",
        "baseline": (existing.get("baseline") if existing else "") or "member_correction",
        "host_version": (existing.get("host_version") if existing else "") or "member_correction",
        "baseline_threshold": 0,
        "text": (
            f"群友 {member} 纠正黑话「{key}」；原释义：{existing['value']}；建议释义：{meaning}"
            if existing
            else f"群友 {member} 补充黑话「{key}」；建议释义：{meaning}"
        ),
        "actor": actor,
        "timestamp": time.time(),
    }
    raw = {
        "category": "jargon",
        "key": key,
        "member": "",
        "value": meaning,
        "reason": f"群友自助提交黑话纠偏（member_correction:{member}）",
        "days": 90,
        "evidence_ids": [ev_id],
    }
    data = candidate(raw, [event], existing)
    data["source_kind"] = "member_correction"
    data["submitted_by"] = actor
    mode = engine.mode("jargon")
    if mode == "llm":
        mode = "llm_then_human"
    proposal = engine.store.propose(data, mode, actor)
    return {
        "ok": True,
        "proposal": proposal,
        "text": f"已提交黑话「{key}」纠偏提案（#{proposal['id'][:8]}），已进入审查队列等待核验。",
    }


def submit_member_habit_correction(
    engine: Any,
    *,
    scope: str,
    member: str,
    note: str,
) -> dict[str, Any]:
    if engine is None or engine.closed:
        return {"ok": False, "text": "群间学习尚未就绪，请稍后再试。"}
    if ":GroupMessage:" not in str(scope or "") or not engine.allowed(scope):
        return {"ok": False, "text": "当前群未启用群间学习。"}
    if engine.is_privacy_muted(scope, member):
        return {"ok": False, "text": "你已开启本群学习静音，请先发送「/学习静音 关闭」后再提交纠偏。"}

    clean_note = " ".join(str(note or "").split())
    if not clean_note or len(clean_note) < 2:
        return {"ok": False, "text": "用法：/习惯纠偏 <说明>（例如：/习惯纠偏 回复我尽量简短一点，不要剧透动漫）"}
    if len(clean_note) > 400:
        return {"ok": False, "text": "纠偏说明请控制在 400 字以内。"}

    limit = _member_correction_limit(engine.config)
    actor = f"member:{member}"
    used_today = engine.store.count_member_corrections_today(scope, actor)
    if used_today >= limit:
        return {"ok": False, "text": f"今日提交纠偏已达上限（{limit} 次/天），请明天再试。"}

    persona = engine.current_personas.get(scope) or engine.shared_persona_id()
    if not persona:
        return {"ok": False, "text": "当前群尚未取得角色快照，请先在群内正常聊天后再提交。"}

    stamp = int(time.time() * 1000)
    ev_id = f"member_correction:habit:{member}:{stamp}"
    event = {
        "id": ev_id,
        "scope": scope,
        "persona": persona,
        "member": member,
        "message_id": ev_id,
        "turn_id": ev_id,
        "kind": "member_correction",
        "baseline": "member_correction",
        "host_version": "member_correction",
        "baseline_threshold": 0,
        "text": clean_note,
        "actor": actor,
        "timestamp": time.time(),
        "context": {
            "source_verified": True,
            "character_scope": engine.character_scopes.get((scope, persona), {}),
        },
    }

    # Check if user explicitly prefixed 边界/不喜欢/喜欢 to create a structured preference proposal
    pref_match = re.match(r"^(边界|不喜欢|讨厌|避雷|喜欢)\s*[:：]?\s*(.+)$", clean_note)
    if pref_match:
        tag, body_text = pref_match.group(1), pref_match.group(2).strip()
        stance = "boundary" if tag == "边界" else "like" if tag == "喜欢" else "dislike"
        topic = body_text[:40]
        raw = {
            "category": "preference",
            "key": topic,
            "member": member,
            "value": {
                "topic": topic,
                "stance": stance,
                "condition": "成员自助纠偏",
                "quote": clean_note,
                "source_id": ev_id,
            },
            "reason": f"群友本人自助提交喜恶/边界纠偏（member_correction:{member}）",
            "days": 60,
            "evidence_ids": [ev_id],
        }
        existing = None
        mode = engine.mode("preference")
    else:
        existing = next(
            (
                e
                for e in engine._entries_for(scope, persona)
                if e.get("category") == "style"
                and e.get("member") == member
                and e.get("key") == "成员自助纠偏"
                and e.get("active")
            ),
            None,
        )
        raw = {
            "category": "style",
            "key": "成员自助纠偏",
            "member": member,
            "value": f"该成员本人自助说明的交流与表达习惯：{clean_note}",
            "reason": f"群友本人自助提交习惯纠偏（member_correction:{member}）",
            "days": 60,
            "evidence_ids": [ev_id],
        }
        mode = engine.mode("style")

    if mode == "llm":
        mode = "llm_then_human"
    data = candidate(raw, [event], existing)
    data["source_kind"] = "member_correction"
    data["submitted_by"] = actor
    proposal = engine.store.propose(data, mode, actor)
    return {
        "ok": True,
        "proposal": proposal,
        "text": f"已收到你的习惯纠偏提案（#{proposal['id'][:8]}），已进入审查队列，核验通过后生效并支持版本回滚。",
    }


async def handle_learning_mute_command(
    engine: Any,
    context: Any,
    *,
    scope: str,
    member: str,
    arg: str = "",
) -> dict[str, Any]:
    if engine is None or engine.closed:
        return {"ok": False, "text": "群间学习尚未就绪，请稍后再试。"}
    if ":GroupMessage:" not in str(scope or ""):
        return {"ok": False, "text": "请在群聊中使用「/学习静音 [开启|关闭]」。"}

    clean = str(arg or "").strip().lower()
    if clean in {"", "状态", "status", "查看"}:
        muted = engine.is_privacy_muted(scope, member)
        state_text = "已开启（本群暂停采集与注入你的学习资料）" if muted else "未开启（正常参与本群学习）"
        return {
            "ok": True,
            "muted": muted,
            "text": f"当前学习静音状态：{state_text}。\n发送「/学习静音 开启」退出采集，发送「/学习静音 关闭」恢复。",
        }

    if clean in {"开启", "开", "on", "true", "1", "mute", "启用"}:
        target = True
    elif clean in {"关闭", "关", "off", "false", "0", "unmute", "恢复", "取消"}:
        target = False
    else:
        return {"ok": False, "text": "用法：/学习静音 [开启|关闭]（留空查看当前状态）"}

    actor = f"member:{member}"
    engine.store.set_local_privacy_mute(scope, member, target, actor=actor)
    synced_core = False
    owner = series_panels.find_plugin_owner(context, series_panels.CHAT_DYNAMICS_ID) if context is not None else None
    setter = getattr(owner, "set_series_privacy_muted", None) if owner is not None else None
    if callable(setter):
        try:
            res = setter(
                scope,
                member,
                target,
                scopes=["learning"],
                actor=actor,
                reason="成员自助学习静音",
            )
            if inspect.isawaitable(res):
                res = await res
            synced_core = isinstance(res, dict) and bool(res.get("ok", True))
        except Exception:
            synced_core = False

    still_muted = engine.is_privacy_muted(scope, member)
    if target:
        sync_note = "（已同步至群间系列隐私静音表）" if synced_core else ""
        return {
            "ok": True,
            "muted": True,
            "synced_core": synced_core,
            "text": f"已为你开启本群学习静音{sync_note}：后续不再采集、提取或投影你的个人学习资料。随时可发「/学习静音 关闭」恢复。",
        }
    if still_muted:
        return {
            "ok": True,
            "muted": True,
            "synced_core": synced_core,
            "text": "已解除本插件的单独学习静音，但你在群间系列中仍处于「全局隐私隐藏」状态；如需全面恢复，请使用「史记恢复 全局」。",
        }
    sync_note = "（已同步至群间系列隐私静音表）" if synced_core else ""
    return {
        "ok": True,
        "muted": False,
        "synced_core": synced_core,
        "text": f"已为你关闭本群学习静音{sync_note}，恢复正常学习与个人印象查询。",
    }
