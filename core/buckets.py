"""One vocabulary for "what went wrong", shared by every layer that names it.

The same failure has to be called the same thing in four places: the
per-sample `error_type` the console counts, the message-level attribution
chain, the capability matrix, and the tuning rules that decide which error an
adjustment was aimed at. Four copies of the strings is four chances to drift,
and a drift here is invisible: the report simply stops adding up.

So the vocabulary is declared once, with the split that actually matters:

```text
model errors      the decision function can be moved to fix them
    recipient_error, topic_candidate_miss, topic_ranking_error, participation_error

system events     the reply did not happen, and no router threshold caused it
    gate_suppression, generation_failure, delivery_failure

no diagnosis      unattributable
none              ok
```

Keeping the two middle groups apart is the whole point of the module. Counting
a 作息压制 as a `missed_reply` makes the router look wrong for a decision it got
right, and it is the one mistake that, once made, cannot be seen in the totals:
the number moves the moment the gate is configured differently, in a plugin
that never touched the router.

This module has no imports on purpose: it is the leaf every other module can
depend on without acquiring a cycle.
"""
from __future__ import annotations

# ---- model errors: a parameter move could plausibly fix these -------------

RECIPIENT_ERROR = "recipient_error"
TOPIC_CANDIDATE_MISS = "topic_candidate_miss"
TOPIC_RANKING_ERROR = "topic_ranking_error"
PARTICIPATION_ERROR = "participation_error"

# ---- system events: the reply did not happen for a non-routing reason -----

GATE_SUPPRESSION = "gate_suppression"
GENERATION_FAILURE = "generation_failure"
DELIVERY_FAILURE = "delivery_failure"

# ---- no diagnosis ---------------------------------------------------------

# The record cannot say which link failed. Kept as a first-class outcome rather
# than folded into `participation_error`: "we do not know" and "the router was
# wrong" are different findings with different fixes.
UNATTRIBUTABLE = "unattributable"
# Nothing in the chain failed.
OK = "ok"

MODEL_ERRORS = (RECIPIENT_ERROR, TOPIC_CANDIDATE_MISS, TOPIC_RANKING_ERROR,
                PARTICIPATION_ERROR)
SYSTEM_EVENTS = (GATE_SUPPRESSION, GENERATION_FAILURE, DELIVERY_FAILURE)

# Report order: the causal chain, then the two catch-alls. `ok` is first so a
# bar chart built from this order reads left-to-right as "nothing wrong" ->
# "wrong earlier in the chain".
ORDER = (OK, *MODEL_ERRORS, *SYSTEM_EVENTS, UNATTRIBUTABLE)
ERROR_ORDER = (*MODEL_ERRORS, *SYSTEM_EVENTS, UNATTRIBUTABLE)

LABEL = {
    OK: "无归因错误",
    RECIPIENT_ERROR: "对话对象判定错误（机器人没被认成收件人）",
    TOPIC_CANDIDATE_MISS: "候选生成缺失（正确话题没进候选集）",
    TOPIC_RANKING_ERROR: "候选排序错误（正确话题在候选集里但没被选中）",
    PARTICIPATION_ERROR: "参与准入错误（该不该进入回复流程判断错了）",
    GATE_SUPPRESSION: "门禁压制（准入正确，但作息/降温/媒体等把发送压掉）",
    GENERATION_FAILURE: "生成失败（已进入回复流程，生成没产出可用回复）",
    DELIVERY_FAILURE: "发送失败（生成了，平台发送失败）",
    UNATTRIBUTABLE: "无法归因（记录不足以判断哪一环出错）",
}

# What a reader should do about each bucket. Written as the fix, not the
# symptom, because "归因" is only worth doing if it changes the next action.
ACTION = {
    OK: "无需处理：这条链上没有发现错误。",
    RECIPIENT_ERROR: "问题在定向：调整 strong_addressivity_threshold 或补充定向证据。",
    TOPIC_CANDIDATE_MISS: "问题在候选生成（embedding / 检索）：改阈值无用。",
    TOPIC_RANKING_ERROR: "问题在打分与排序：consider topic_commit_threshold / topic_margin_threshold。",
    PARTICIPATION_ERROR: "问题在参与准入评分：检查证据权重与阈值。",
    GATE_SUPPRESSION: "不是路由问题：门禁（作息/降温/媒体）压掉了发送，本插件只能观测。",
    GENERATION_FAILURE: "不是路由问题：生成阶段失败，先看模型与超时配置。",
    DELIVERY_FAILURE: "不是路由问题：平台发送失败，先看适配器与限流。",
    UNATTRIBUTABLE: "记录不足：需要本体补齐候选集或最终结果才能归因。",
}


def label_for(bucket: str) -> str:
    return LABEL.get(bucket, bucket)


def is_model_error(bucket: str) -> bool:
    return bucket in MODEL_ERRORS


__all__ = [
    "ACTION", "DELIVERY_FAILURE", "ERROR_ORDER", "GATE_SUPPRESSION", "GENERATION_FAILURE",
    "LABEL", "MODEL_ERRORS", "OK", "ORDER", "PARTICIPATION_ERROR", "RECIPIENT_ERROR",
    "SYSTEM_EVENTS", "TOPIC_CANDIDATE_MISS", "TOPIC_RANKING_ERROR", "UNATTRIBUTABLE",
    "is_model_error", "label_for",
]
