"""Declarative operational presets and review mode validation for Dynamics Learning."""

from __future__ import annotations

from typing import Any

from .contract import CATEGORIES, MODES

SCENE_EXPRESSION_CATEGORIES = frozenset(
    {"scene_pattern", "scene_adjustment", "expression_style", "expression_adjustment"}
)

OPERATION_PRESETS: dict[str, dict[str, Any]] = {
    "conservative": {
        "id": "conservative",
        "label": "省钱保守",
        "summary": "低频分析、关闭试验模块、以人工复核为主，显著降低模型调用成本。",
        "config": {
            "analysis_interval": 1200,
            "batch_events": 35,
            "analysis_max_rounds_per_cycle": 1,
            "automatic_review_batch_enabled": True,
            "scene_learning_enabled": False,
            "scene_disambiguation_enabled": False,
            "expression_style_enabled": False,
            "style_review": "llm_then_human",
            "typing_habit_review": "llm_then_human",
            "preference_review": "llm_then_human",
            "expression_example_review": "llm_then_human",
            "jargon_review": "human",
            "relationship_review": "human",
            "state_review": "human",
            "decision_guidance_review": "human",
            "participation_review": "human",
            "affection_review": "llm_then_human",
        },
    },
    "balanced": {
        "id": "balanced",
        "label": "平衡活人感",
        "summary": "推荐默认：成员习惯与喜恶经 LLM 审查自动生效，好感带护栏自动更新，关键策略人工把关。",
        "config": {
            "analysis_interval": 600,
            "batch_events": 20,
            "analysis_max_rounds_per_cycle": 2,
            "automatic_review_batch_enabled": True,
            "scene_learning_enabled": False,
            "scene_disambiguation_enabled": False,
            "expression_style_enabled": False,
            "style_review": "llm",
            "typing_habit_review": "llm",
            "preference_review": "llm",
            "expression_example_review": "llm",
            "jargon_review": "llm_then_human",
            "relationship_review": "llm_then_human",
            "state_review": "llm_then_human",
            "decision_guidance_review": "llm_then_human",
            "participation_review": "llm_then_human",
            "affection_review": "auto",
        },
    },
    "deep_resonance": {
        "id": "deep_resonance",
        "label": "深度共鸣",
        "summary": "高频捕捉群聊节奏、开启场景学习与群体表达习惯，黑话与短期状态经 LLM 自动生效。",
        "config": {
            "analysis_interval": 300,
            "batch_events": 12,
            "analysis_max_rounds_per_cycle": 3,
            "automatic_review_batch_enabled": True,
            "scene_learning_enabled": True,
            "scene_disambiguation_enabled": False,
            "expression_style_enabled": True,
            "style_review": "llm",
            "typing_habit_review": "llm",
            "preference_review": "llm",
            "expression_example_review": "llm",
            "jargon_review": "llm",
            "relationship_review": "llm_then_human",
            "state_review": "llm",
            "decision_guidance_review": "llm_then_human",
            "participation_review": "llm_then_human",
            "affection_review": "auto",
            "scene_pattern_review": "llm",
            "scene_adjustment_review": "llm",
            "expression_style_review": "llm",
            "expression_adjustment_review": "llm",
        },
    },
}


def allowed_modes_for_category(category: str) -> tuple[str, ...]:
    if category in SCENE_EXPRESSION_CATEGORIES:
        return ("llm", "llm_then_human")
    if category == "affection":
        return ("auto", "human", "llm", "llm_then_human")
    return ("human", "llm", "llm_then_human")


def validate_review_modes(updates: Any) -> dict[str, str]:
    if not isinstance(updates, dict) or not updates:
        raise ValueError("请提供需要修改的审查方式")
    cleaned: dict[str, str] = {}
    for raw_cat, raw_mode in updates.items():
        cat = str(raw_cat or "").strip()
        mode = str(raw_mode or "").strip()
        if cat not in CATEGORIES:
            raise ValueError(f"未知学习类别：{cat}")
        if mode not in MODES or mode not in allowed_modes_for_category(cat):
            raise ValueError(f"类别 {cat} 不支持审查方式 {mode}")
        cleaned[cat] = mode
    return cleaned


def detect_active_preset(config: dict[str, Any]) -> str:
    for preset_id, spec in OPERATION_PRESETS.items():
        target = spec["config"]
        if all(config.get(k) == v for k, v in target.items()):
            return preset_id
    return "custom"


def preset_catalog() -> list[dict[str, Any]]:
    return [
        {"id": spec["id"], "label": spec["label"], "summary": spec["summary"]}
        for spec in OPERATION_PRESETS.values()
    ]
