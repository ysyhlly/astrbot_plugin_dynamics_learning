"""Shared config normalization, extended inheritance, and cross-plugin privacy mute helpers."""

from __future__ import annotations

import hashlib
import inspect
import re
from typing import Any

from .. import series_panels

_UNBOUND_PERSONA_SENTINELS = frozenset({"__none__", "[%None]"})
_SPLIT_RE = re.compile(r"[\r\n,，;；]+")


def normalize_str_list(raw: Any) -> list[str]:
    """Normalize a list/tuple/set OR comma/newline-separated text into deduplicated strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items: list[Any] = _SPLIT_RE.split(raw)
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, bool):
            continue
        if isinstance(item, (str, int)):
            clean = str(item).strip()
            if clean and clean not in seen:
                seen.add(clean)
                out.append(clean)
    return out


def _read_raw_shared_config(context: Any) -> dict[str, Any]:
    if context is None:
        return {}
    owner = series_panels.find_plugin_owner(context, series_panels.CHAT_DYNAMICS_ID)
    if owner is None:
        return {}
    getter = getattr(owner, "get_series_shared_config", None)
    if not callable(getter) or inspect.iscoroutinefunction(getter):
        return {}
    try:
        raw = getter()
    except Exception:
        return {}
    if inspect.isawaitable(raw):
        if inspect.iscoroutine(raw):
            raw.close()
        return {}
    if not isinstance(raw, dict) or not raw or raw.get("enabled") is False:
        return {}
    return raw


def read_extended_shared_config(context: Any) -> dict[str, Any]:
    """Read series shared config including Wave 1 extended slots (read-only, sync).

    Combines ``series_panels.get_series_shared_config(context)`` with extended Wave 1
    keys (``light_analysis_provider``, ``vision_provider``, ``embedding_provider``,
    ``rerank_provider``, ``private_shared_group_sources``) so companion code works both
    before and after ``series_panels.py`` is synced.
    """
    if context is None:
        return {}
    base = series_panels.get_series_shared_config(context)
    result: dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    raw = _read_raw_shared_config(context)
    if not raw:
        if "private_shared_group_sources" in result:
            result["private_shared_group_sources"] = normalize_str_list(
                result.get("private_shared_group_sources")
            )
        return result

    for str_key in (
        "light_analysis_provider",
        "vision_provider",
        "embedding_provider",
        "rerank_provider",
        "persona_id",
        "reply_provider",
        "decision_provider",
    ):
        if not result.get(str_key):
            val = raw.get(str_key)
            if isinstance(val, str) and val.strip():
                result[str_key] = val.strip()

    if "private_shared_group_sources" in raw or "private_shared_group_sources" in result:
        raw_sources = (
            result.get("private_shared_group_sources")
            if result.get("private_shared_group_sources")
            else raw.get("private_shared_group_sources")
        )
        result["private_shared_group_sources"] = normalize_str_list(raw_sources)
    else:
        result["private_shared_group_sources"] = []
    return result


def resolve_shared_persona_id(context: Any) -> str:
    """Return the inherited shared persona_id from chat_dynamics, ignoring unbound sentinels."""
    shared = read_extended_shared_config(context)
    raw = str(shared.get("persona_id") or "").strip()[:256]
    if not raw or raw in _UNBOUND_PERSONA_SENTINELS:
        return ""
    return raw


def is_member_privacy_muted(
    context: Any, umo: str, member_id: str, scope: str = "learning"
) -> bool:
    """Synchronous cross-plugin privacy mute check against chat_dynamics (fail-open False)."""
    clean_umo = str(umo or "").strip()
    clean_member = str(member_id or "").strip()
    clean_scope = str(scope or "learning").strip() or "learning"
    if context is None or not clean_umo or not clean_member:
        return False

    owner = series_panels.find_plugin_owner(context, series_panels.CHAT_DYNAMICS_ID)
    if owner is not None:
        getter = getattr(owner, "get_series_privacy_muted", None)
        if callable(getter) and not inspect.iscoroutinefunction(getter):
            try:
                res = getter(clean_umo, clean_member, clean_scope)
            except TypeError:
                try:
                    res = getter(clean_umo, clean_member)
                except Exception:
                    return False
            except Exception:
                return False
            if inspect.isawaitable(res):
                if inspect.iscoroutine(res):
                    res.close()
                return False
            if isinstance(res, bool):
                return res
            if isinstance(res, dict):
                return bool(res.get("muted", False))
            return False

    helper = getattr(series_panels, "is_member_privacy_muted", None)
    if callable(helper) and not inspect.iscoroutinefunction(helper):
        try:
            res = helper(context, clean_umo, clean_member, clean_scope)
            if inspect.isawaitable(res):
                if inspect.iscoroutine(res):
                    res.close()
                return False
            return bool(res)
        except Exception:
            return False
    return False


def _add_persona_label(labels: dict[str, str], key: str, display: str) -> None:
    clean_k = str(key or "").strip()
    clean_v = str(display or "").strip()[:64]
    if not clean_k or not clean_v or clean_k in _UNBOUND_PERSONA_SENTINELS or clean_v in _UNBOUND_PERSONA_SENTINELS:
        return
    if clean_k not in labels:
        labels[clean_k] = clean_v
    if len(clean_k) >= 12 and clean_k[:12] not in labels:
        labels[clean_k[:12]] = clean_v


def resolve_persona_labels(context: Any, store: Any = None, engine: Any = None) -> dict[str, str]:
    """Map persona SHA-256 hashes and IDs to human-readable character/persona names."""
    labels: dict[str, str] = {}

    # 1. From live engine character_scopes & store native_scope_registry
    char_scopes: dict[tuple[str, str], dict[str, Any]] = {}
    if engine is not None and isinstance(getattr(engine, "character_scopes", None), dict):
        char_scopes.update(engine.character_scopes)
    if store is not None and hasattr(store, "native"):
        try:
            for pair, body in (store.native.scopes() or {}).items():
                char_scopes.setdefault(pair, body)
        except Exception:
            pass

    for (_scope, persona_key), body in char_scopes.items():
        if not isinstance(body, dict):
            continue
        pid = str(body.get("persona_id") or body.get("name") or "").strip()
        if pid:
            _add_persona_label(labels, persona_key, pid)
            _add_persona_label(labels, pid, pid)

    # 2. From AstrBot host persona_manager (matches both persona_id and sha256(prompt))
    pm = getattr(context, "persona_manager", None) if context is not None else None
    raw_personas: list[Any] = []
    if pm is not None:
        for attr in ("personas", "personas_v3"):
            items = getattr(pm, attr, None)
            if isinstance(items, (list, tuple)):
                raw_personas.extend(items)
            elif isinstance(items, dict):
                raw_personas.extend(items.values())
    for item in raw_personas:
        if isinstance(item, dict):
            pid = str(item.get("persona_id") or item.get("name") or item.get("id") or "").strip()
            name = str(item.get("name") or pid).strip()
            prompt = str(item.get("system_prompt") or item.get("prompt") or "").strip()
        else:
            pid = str(getattr(item, "persona_id", "") or getattr(item, "name", "") or "").strip()
            name = str(getattr(item, "name", "") or pid).strip()
            prompt = str(getattr(item, "system_prompt", "") or getattr(item, "prompt", "") or "").strip()
        if not name:
            continue
        if pid:
            _add_persona_label(labels, pid, name)
            _add_persona_label(labels, hashlib.sha256(pid.encode("utf-8")).hexdigest(), name)
        if prompt:
            _add_persona_label(labels, hashlib.sha256(prompt.encode("utf-8")).hexdigest(), name)

    # 3. From dynamics_world companion if active (optional display name enrichment)
    if context is not None:
        world = series_panels.find_plugin_owner(context, "astrbot_plugin_dynamics_world")
        if world is not None:
            for method_name in ("list_series_personas", "get_character_names"):
                fn = getattr(world, method_name, None)
                if callable(fn) and not inspect.iscoroutinefunction(fn):
                    try:
                        raw = fn()
                        if inspect.isawaitable(raw):
                            if inspect.iscoroutine(raw):
                                raw.close()
                            continue
                        if isinstance(raw, dict):
                            for k, v in raw.items():
                                _add_persona_label(labels, str(k), str(v))
                        elif isinstance(raw, (list, tuple)):
                            for row in raw:
                                if isinstance(row, dict):
                                    pid = str(row.get("persona_id") or row.get("id") or "").strip()
                                    name = str(row.get("name") or row.get("display_name") or pid).strip()
                                    if pid and name:
                                        _add_persona_label(labels, pid, name)
                    except Exception:
                        continue

    # 4. From shared persona_id as fallback label for itself
    shared_pid = resolve_shared_persona_id(context)
    if shared_pid:
        _add_persona_label(labels, shared_pid, shared_pid)
    return labels
