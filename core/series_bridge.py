"""Shared config normalization, extended inheritance, and cross-plugin privacy mute helpers."""

from __future__ import annotations

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
