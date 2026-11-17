"""Bounded interaction and member-balanced network metrics.

The daily observations contain counts and tone totals, never message text.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from typing import Any

TZ = dt.timezone(dt.timedelta(hours=8))


def day_key(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, TZ).date().isoformat()


def window_start(ts: int, days: int) -> str:
    return (
        dt.datetime.fromtimestamp(ts, TZ).date() - dt.timedelta(days=days - 1)
    ).isoformat()


def bounded_interaction(evidence: float, scale: float = 120.0) -> float:
    evidence = max(0.0, float(evidence))
    return 100.0 * evidence / (evidence + max(1.0, scale))


def signed_affinity(positive: float, negative: float) -> float:
    positive, negative = max(0.0, positive), max(0.0, negative)
    return 100.0 * (positive - negative) / (positive + negative + 6.0)


def activity_observation(
    buckets: list[dict[str, Any]], *, now: int, window: int = 30, daily_cap: int = 80
) -> dict[str, Any]:
    """Count sustained activity with a daily volume ceiling.

    Args:
        buckets: Member daily counts, including optional estimated flags.
        now: Current Unix timestamp.
        window: Observation window in calendar days.
        daily_cap: Maximum daily count used for activity scoring.

    Returns:
        Observed days, recency-weighted days and capped message volume.
    """
    today = dt.datetime.fromtimestamp(now, TZ).date()
    active_days = estimated_days = 0
    weighted_days = effective_messages = 0.0
    for bucket in buckets:
        age = (today - dt.date.fromisoformat(bucket["day"])).days
        count = max(0, int(bucket.get("message_count", 0)))
        if not 0 <= age < window or not count:
            continue
        weight = 0.5 ** (age / max(1.0, window / 2))
        active_days += 1
        estimated_days += int(bool(bucket.get("estimated", 0)))
        weighted_days += weight
        effective_messages += min(count, max(1, daily_cap)) * weight
    return {
        "active_days": active_days,
        "weighted_active_days": weighted_days,
        "effective_messages": effective_messages,
        "activity_estimated_days": estimated_days,
    }


def tone_state(item: dict[str, Any]) -> str:
    """Separate missing evidence from an observed neutral tendency.

    Args:
        item: Pair or member metrics with tone confidence and tendencies.

    Returns:
        unknown, neutral, positive, negative, or mixed.
    """
    if float(item.get("affinity_confidence", 0)) < 0.2:
        return "unknown"
    affinity = float(item.get("affinity", 0))
    negative = float(
        item.get("negative_tendency", item.get("conflict_risk", max(0, -affinity)))
    )
    positive = float(item.get("positive_tendency", max(0, affinity + negative)))
    if positive >= 20 and negative >= 20:
        return "mixed"
    if affinity >= 20:
        return "positive"
    if affinity <= -20:
        return "negative"
    return "neutral"


def pair_metrics(
    row: dict[str, Any],
    days: list[dict[str, Any]],
    *,
    now: int,
    window: int = 30,
    scale: float = 120.0,
) -> dict[str, Any]:
    """Convert daily observations to a bounded, recent-weighted pair summary."""
    today = dt.datetime.fromtimestamp(now, TZ).date()
    recent = defaultdict(float)
    active_days = 0
    observed_tone_days = 0
    for bucket in days:
        age = (today - dt.date.fromisoformat(bucket["day"])).days
        if not 0 <= age < window:
            continue
        weight = 0.5 ** (age / max(1.0, window / 2.0))
        active_days += int(float(bucket.get("evidence", 0)) > 0)
        observed_tone_days += int(bucket.get("sentiment_hits", 0) > 0)
        for field in (
            "evidence",
            "friendly",
            "hostile",
            "ab_friendly",
            "ab_hostile",
            "ba_friendly",
            "ba_hostile",
        ):
            recent[field] += float(bucket.get(field, 0) or 0) * weight
        for field in (
            "raw_evidence",
            "hits",
            "focused_hits",
            "directed_hits",
            "reply_hits",
            "at_hits",
            "ab_hits",
            "ba_hits",
            "sentiment_hits",
        ):
            recent[field] += float(bucket.get(field, 0) or 0)
    # Old totals are only a small, capped memory prior, never fresh evidence.
    history = (
        min(scale, max(0.0, float(row.get("score", 0) or 0) - recent["raw_evidence"]))
        * 0.15
    )
    evidence = recent["evidence"] + history
    score = bounded_interaction(evidence, scale)
    recent_score = bounded_interaction(recent["evidence"], scale)
    ab, ba = recent["ab_hits"], recent["ba_hits"]
    reciprocity = 2.0 * min(ab, ba) / (ab + ba) if ab + ba else 0.0
    tier = "directed" if recent["directed_hits"] else ("ambient" if days else "legacy")
    confidence = min(1.0, recent["evidence"] / 80.0) * min(1.0, active_days / 5.0)
    confidence *= (0.4 + 0.6 * reciprocity) if tier == "directed" else 0.2
    tone_total = recent["friendly"] + recent["hostile"]
    affinity_confidence = min(1.0, tone_total / 12.0) * min(
        1.0, observed_tone_days / 3.0
    )
    affinity = signed_affinity(recent["friendly"], recent["hostile"])
    positive_tendency = 100.0 * recent["friendly"] / (tone_total + 6.0)
    negative_tendency = 100.0 * recent["hostile"] / (tone_total + 6.0)
    state = tone_state(
        {
            "affinity_confidence": affinity_confidence,
            "affinity": affinity,
            "positive_tendency": positive_tendency,
            "negative_tendency": negative_tendency,
        }
    )
    tone = {
        "unknown": "待观察",
        "positive": "偏友善",
        "negative": "偏冲突",
        "mixed": "正负并存",
        "neutral": "整体中性",
    }[state]
    if score >= 80 and active_days >= 10 and reciprocity >= 0.35 and confidence >= 0.6:
        label = "稳定强互动"
    elif score >= 50 and active_days >= 3:
        label = "持续互动"
    elif score >= 15:
        label = "有来有往"
    else:
        label = "初步接触" if recent["evidence"] else "历史参考"
    if not recent["directed_hits"] and not recent["focused_hits"]:
        label = "环境共现" if recent["evidence"] else "历史参考"
    return {
        **row,
        "algorithm_version": 2,
        "score": score,
        "raw_score": float(row.get("score", 0) or 0),
        "evidence_value": evidence,
        "recent_score": recent_score,
        "recent_evidence": recent["evidence"],
        "historical_evidence": history,
        "friendly_score": recent["friendly"],
        "hostile_score": recent["hostile"],
        "affinity": affinity,
        "affinity_confidence": affinity_confidence,
        "positive_tendency": positive_tendency,
        "negative_tendency": negative_tendency,
        "a_to_b_affinity": signed_affinity(recent["ab_friendly"], recent["ab_hostile"]),
        "b_to_a_affinity": signed_affinity(recent["ba_friendly"], recent["ba_hostile"]),
        "tone": tone,
        "tone_state": state,
        "evidence_tier": tier,
        "confidence": confidence,
        "active_days": active_days,
        "reciprocity": reciprocity,
        "strength_label": label,
        **{
            field: int(recent[field])
            for field in (
                "hits",
                "focused_hits",
                "directed_hits",
                "reply_hits",
                "at_hits",
                "ab_hits",
                "ba_hits",
                "sentiment_hits",
            )
        },
    }


def network_metrics(
    users: dict[str, Any], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize all pairs before display limits; weight members equally."""
    contacts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        # Ambient noise and history alone do not imply a current social contact.
        if (
            pair.get("recent_score", pair["score"]) >= 5
            and pair.get("directed_hits", 0) + pair.get("focused_hits", 0) > 0
        ):
            contacts[str(pair["a"])].append(pair)
            contacts[str(pair["b"])].append(pair)
    profiles = []
    for uid, user in users.items():
        edges = contacts[uid]
        weights = [min(1.0, e["score"] / 30.0) for e in edges]
        total = sum(weights)
        effective = total * total / sum(w * w for w in weights) if total else 0.0
        breadth = 100.0 * (1 - math.exp(-total / 4.0))
        supported_effective = effective * total / len(edges) if edges else 0.0
        balance = 100.0 * (1 - math.exp(-supported_effective / 4.0))
        mean_strength = sum(e["score"] for e in edges) / len(edges) if edges else 0.0
        reciprocal = sum(e["reciprocity"] for e in edges) / len(edges) if edges else 0.0
        tone_weights = [w * e["affinity_confidence"] for w, e in zip(weights, edges)]
        tone_mass = sum(tone_weights)
        affinity = (
            sum(e["affinity"] * w for e, w in zip(edges, tone_weights)) / tone_mass
            if tone_mass
            else 0.0
        )
        conflict = (
            sum(
                e.get("negative_tendency", max(0.0, -e["affinity"])) * w
                for e, w in zip(edges, tone_weights)
            )
            / tone_mass
            if tone_mass
            else 0.0
        )
        positive = sum(
            e.get("positive_tendency", e["affinity"]) >= 20
            and e["affinity_confidence"] >= 0.2
            for e in edges
        )
        negative = sum(
            e.get("negative_tendency", -e["affinity"]) >= 20
            and e["affinity_confidence"] >= 0.2
            for e in edges
        )
        mixed = sum(
            e.get("positive_tendency", 0) >= 20
            and e.get("negative_tendency", 0) >= 20
            and e["affinity_confidence"] >= 0.2
            for e in edges
        )
        count = max(0, int(user.get("count", 0)))
        weighted_days = float(user.get("weighted_active_days", int(count > 0)))
        effective_messages = float(user.get("effective_messages", min(count, 80)))
        cap = max(1, int(user.get("activity_daily_cap", 80)))
        activity_score = (
            45 * (1 - math.exp(-weighted_days / 5))
            + 35 * (1 - math.exp(-effective_messages / (cap * 4)))
            + breadth * 0.20
        )
        profiles.append(
            {
                "user_id": uid,
                "name": user.get("name", uid),
                "partner_count": len(edges),
                "effective_partners": round(effective, 2),
                "breadth": breadth,
                "balance": balance,
                "reciprocity": reciprocal,
                "mean_interaction": mean_strength,
                "social_score": breadth * 0.4
                + balance * 0.25
                + mean_strength * 0.2
                + reciprocal * 15,
                "affinity": affinity,
                "conflict_risk": conflict,
                "positive_contacts": positive,
                "negative_contacts": negative,
                "mixed_contacts": mixed,
                "unknown_contacts": sum(e["affinity_confidence"] < 0.2 for e in edges),
                "affinity_confidence": min(1.0, tone_mass / max(1, len(edges))),
                "dominant_share": max(weights) / total if total else 0.0,
                "message_count": user.get("count", 0),
                "active_days": int(user.get("active_days", int(count > 0))),
                "effective_messages": effective_messages,
                "activity_score": activity_score,
                "activity_estimated_days": int(
                    user.get("activity_estimated_days", int(count > 0))
                ),
            }
        )
        profiles[-1]["tone_state"] = tone_state(profiles[-1])
    active = [p for p in profiles if p["message_count"] > 0 or p["partner_count"] > 0]
    known = [p for p in active if p["affinity_confidence"] >= 0.2]

    def mean(field):
        return sum(p[field] for p in active) / len(active) if active else 0.0

    # Unobserved tone keeps a neutral prior and a visible low confidence;
    # a few strongly positive members cannot speak for the whole active group.
    affinity = sum(p["affinity"] for p in known) / len(active) if active else 0.0
    conflict = sum(p["conflict_risk"] for p in known) / len(active) if active else 0.0
    profiles.sort(key=lambda p: p["social_score"], reverse=True)
    eligible = {
        (str(e["a"]), str(e["b"])): e for edges in contacts.values() for e in edges
    }
    known_pairs = sum(e["affinity_confidence"] >= 0.2 for e in eligible.values())
    return {
        "member_profiles": profiles,
        "active_member_count": len(active),
        "connected_member_count": sum(p["partner_count"] > 0 for p in active),
        "breadth": mean("breadth"),
        "balance": mean("balance"),
        "reciprocity": mean("reciprocity") * 100,
        "affinity": affinity,
        "conflict_risk": conflict,
        "tone_member_count": len(known),
        "affinity_confidence": sum(p["affinity_confidence"] for p in active)
        / len(active)
        if active
        else 0.0,
        "tone_pair_count": known_pairs,
        "tone_pair_total": len(eligible),
        "tone_member_coverage": 100 * len(known) / len(active) if active else 0.0,
        "tone_pair_coverage": 100 * known_pairs / len(eligible) if eligible else 0.0,
    }
