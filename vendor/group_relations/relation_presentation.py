"""Representative graph edges; display choices never change network scores."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def relation_explanations(
    summary: dict[str, Any], *, window_days: int = 30
) -> dict[str, Any]:
    """Explain whole-group participation, distribution and tone evidence.

    Args:
        summary: Full-network metrics, before graph display limits.
        window_days: Calendar-day observation window used by the metrics.

    Returns:
        Whole-group paragraphs without individual members or pair notes.
    """
    overview = []
    if "active_member_count" not in summary:
        overview.append("暂无完整群聊统计，暂不生成全群关系判断。")
    else:
        active = int(summary.get("active_member_count") or 0)
        connected = int(summary.get("connected_member_count") or 0)
        if not connected:
            overview.append(
                f"近{window_days}天观察到{active}位成员，尚无足够明确的有效互动，暂不判断关系分布。"
            )
        else:
            overview.append(
                f"近{window_days}天观察到{active}位成员，其中{connected}位已建立有效互动。"
            )
            breadth = float(summary.get("breadth") or 0)
            reach = (
                "与不同伙伴的交流较广泛"
                if breadth >= 65
                else "与不同伙伴的有效交流整体较少"
                if breadth < 35
                else "与不同伙伴已有一定交流"
            )
            balance = float(summary.get("balance") or 0)
            distribution = (
                "互动分布较分散"
                if balance >= 65
                else "互动较集中在少数伙伴之间"
                if balance < 35
                else "互动均衡处于中等水平"
            )
            reciprocity = float(summary.get("reciprocity") or 0)
            exchange = (
                "双方来回较多"
                if reciprocity >= 65
                else "往来更偏向一方发起"
                if reciprocity < 35
                else "已有一定双向往来"
            )
            overview.append(f"全群{reach}，{distribution}；{exchange}。")
    if float(summary.get("affinity_confidence") or 0) < 0.2:
        overview.append("整体正负语气仍待观察；频繁交流暂不能作为关系友好的依据。")
    else:
        affinity = float(summary.get("affinity") or 0)
        negative = float(summary.get("conflict_risk") or 0)
        if negative >= 20 and affinity + negative >= 20:
            overview.append("现有语气中，友善与冲突迹象同时存在，需要分别看待。")
        elif negative >= 20:
            overview.append("群聊已有较多负向语气迹象，整体沟通氛围需继续观察。")
        elif affinity >= 20:
            overview.append("已有语气整体偏友善，结论仅覆盖已观察到的互动。")
        else:
            overview.append("已有语气整体接近中性，暂未呈现明显的正负倾向。")

    return {"overview": overview}


def representative_relations(
    pairs: list[dict[str, Any]], node_ids: set[str], per_member: int = 3
) -> list[dict[str, Any]]:
    """Select a sparse graph while keeping coverage and confirmed negative ties.

    Args:
        pairs: Complete current pair summaries, before ranking limits.
        node_ids: Members visible in the graph.
        per_member: Ordinary edge target per member, between two and three.

    Returns:
        Selected original rows. Coverage and negative ties may exceed the target
        for a hub; omitted edges remain available in statistics and exports.
    """
    candidates = {}
    for pair in pairs:
        a, b = str(pair["a"]), str(pair["b"])
        if a not in node_ids or b not in node_ids or a == b:
            continue
        if pair.get("algorithm_version") == 2 and not (
            pair.get("recent_score", 0) >= 5
            and pair.get("directed_hits", 0) + pair.get("focused_hits", 0) > 0
        ):
            continue
        candidates[tuple(sorted((a, b)))] = pair
    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            float(item[1].get("score", 0))
            * (0.4 + 0.6 * float(item[1].get("confidence", 0.5))),
            float(item[1].get("reciprocity", 0)),
            item[0],
        ),
        reverse=True,
    )
    selected = {}
    degrees: dict[str, int] = defaultdict(int)
    for key, pair in ranked:
        if (
            float(pair.get("affinity_confidence", 0)) >= 0.2
            and float(
                pair.get("negative_tendency", max(0, -float(pair.get("affinity", 0))))
            )
            >= 20
        ):
            selected[key] = pair
            for uid in key:
                degrees[uid] += 1
    # A maximum-strength forest keeps every original connected component
    # connected. Coverage/bridges can exceed the ordinary target for a hub.
    parents = {uid: uid for uid in node_ids}
    for key, pair in ranked:
        left, right = key
        while parents[left] != left:
            left = parents[left]
        while parents[right] != right:
            right = parents[right]
        if left == right:
            continue
        parents[left] = right
        if key not in selected:
            selected[key] = pair
            for endpoint in key:
                degrees[endpoint] += 1
    target = max(2, min(3, int(per_member)))
    for key, pair in ranked:
        if key in selected or any(degrees[uid] >= target for uid in key):
            continue
        selected[key] = pair
        for uid in key:
            degrees[uid] += 1
    return sorted(
        selected.values(), key=lambda pair: float(pair.get("score", 0)), reverse=True
    )
