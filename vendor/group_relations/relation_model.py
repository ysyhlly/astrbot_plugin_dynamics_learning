"""Reuse WifePicker's live GPU worker; never install or own a second model."""

import asyncio
import math
import re


def model_tone(text, probabilities, fallback):
    """Bounded, person-directed evidence; no duplicate lexical/model credit."""
    if not probabilities:
        return fallback
    try:
        p, n, h = (float(probabilities[k]) for k in ("positive", "neutral", "negative"))
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (p, n, h)) or abs(p+n+h-1) > .02:
            return fallback
    except (KeyError, TypeError, ValueError):
        return fallback
    # A directed message can still be about software, weather, etc. Abstain.
    if not re.search(r"你|您|\byou\b|\byour\b", text, re.I):
        return fallback
    if re.search(r"游戏|软件|代码|网络|天气|服务器|工作|饭|电影|新闻|他说|她说|听说|[“”\"]", text):
        return fallback
    if re.search(r"哈+|笑死|doge|233|hhh|反话|开玩笑", text, re.I):
        return fallback
    friendly, hostile, tags = fallback
    if p >= .8 and h <= .1 and hostile == 0:
        return max(friendly, p), hostile, list(dict.fromkeys([*tags, "友善", "GPU情感"]))
    if h >= .85 and p <= .1 and friendly == 0:
        return friendly, max(hostile, h), list(dict.fromkeys([*tags, "敌对", "GPU情感"]))
    return fallback


class SharedSentiment:
    def __init__(self, context, logger):
        self.context = context
        self.logger = logger
        self.pending = None
        self.announced = False

    async def classify(self, text):
        if self.pending is not None and not self.pending.done():
            return None
        try:
            meta = self.context.get_registered_star("astrbot_plugin_wifepicker")
            if not meta or not meta.activated:
                return None
            owner = meta.star_cls
            config = owner.config.get("group_affinity", {})
            if not config.get("enabled") or not config.get("use_small_model"):
                return None
            model = owner.affinity.model
            if not model.ready or model.lock.locked():
                return None
            self.pending = asyncio.create_task(model.classify(text[:512]))
            # Caller cancellation must not interrupt the shared JSON stream.
            self.pending.add_done_callback(self._consume_exception)
            result = await asyncio.shield(self.pending)
            if result and not self.announced:
                self.logger.info("[group_relation.model] 已复用今日老婆 GPU 情感模型，首次推理成功")
                self.announced = True
            return result
        except Exception as exc:
            self.logger.debug(f"[group_relation.model] 使用规则语气: {type(exc).__name__}")
            return None

    @staticmethod
    def _consume_exception(task):
        if not task.cancelled():
            task.exception()

    async def close(self):
        # Do not terminate a model owned by another plugin.
        if self.pending is not None:
            await asyncio.gather(asyncio.shield(self.pending), return_exceptions=True)
