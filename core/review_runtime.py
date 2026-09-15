"""Review request coordination and cache identity for the plugin orchestrator."""
from __future__ import annotations
import asyncio
import hashlib
import inspect
import json
import sys
import time
from typing import Any


class ReviewRuntimeMixin:
    def _init_review_runtime(self):
        self._review_epoch = 0
        self._review_tasks = {}
        self._review_slots = asyncio.Semaphore(2)
        self._review_write_lock = asyncio.Lock()

    def _review_symbols(self):
        return sys.modules[type(self).__module__]

    async def invalidate_reviews(self):
        if not hasattr(self, "_review_epoch"):
            self._init_review_runtime()
        async with self._review_write_lock:
            self._review_epoch += 1
            tasks = list(self._review_tasks.values())
            self._review_tasks.clear()
            self._reply_review_cache = {}
            self._review_failed_at = self._reply_review_failed_at = 0.0
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _save_review_if_current(self, epoch, value):
        async with self._review_write_lock:
            if epoch == self._review_epoch and self.runtime_config().review_enabled:
                await self.store.save_review(value)
                return True
            return False

    async def _review_current(self, epoch, field, provider_id, model):
        config = self.runtime_config()
        if getattr(self, "_shutting_down", False) or epoch != self._review_epoch or not getattr(config, field + "_enabled"):
            return False
        current = await self._review_provider(getattr(config, field + "_provider_id"))
        return (epoch == self._review_epoch and current == (provider_id, model)
                and not getattr(self, "_shutting_down", False)
                and getattr(self.runtime_config(), field + "_enabled"))

    def _review_identity(self, kind, fingerprint, provider, model, digest, messages=None):
        symbols = self._review_symbols()
        prompt = (symbols.build_prompt if kind == "contract" else symbols.build_reply_prompt)(digest)
        schema = symbols.REVIEW_SCHEMA_VERSION if kind == "contract" else symbols.REPLY_REVIEW_SCHEMA_VERSION
        parser = symbols.parse_review if kind == "contract" else symbols.parse_reply_review
        try:
            parser_version = hashlib.sha256(inspect.getsource(parser).encode()).hexdigest()
        except (TypeError, OSError):
            parser_version = str(schema)
        identity = [kind, fingerprint, provider, model, prompt, schema, parser_version, messages]
        return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

    async def _review_singleflight(self, kind, **kwargs):
        if not hasattr(self, "_review_epoch"):
            self._init_review_runtime()
        epoch = self._review_epoch
        config = self.runtime_config()
        field = "review" if kind == "contract" else "reply_review"
        if getattr(self, "_shutting_down", False):
            return self._review_unavailable(kind)
        provider = (await self._review_provider(getattr(config, field + "_provider_id"))
                    if getattr(config, field + "_enabled") else ("", ""))
        if epoch != self._review_epoch or getattr(self, "_shutting_down", False):
            return self._review_unavailable(kind)
        key = (kind, repr(config), epoch, provider,
               id(kwargs.get("sp_module")) if kind == "reply" else None)

        task = self._review_tasks.get(key)
        if task is None:
            async def run():
                async with self._review_slots:
                    if epoch != self._review_epoch or getattr(self, "_shutting_down", False):
                        return self._review_unavailable(kind)
                    method = self._contract_review_payload_impl if kind == "contract" else self._reply_review_payload_impl
                    return await method(**kwargs)
            task = asyncio.create_task(run())
            self._review_tasks[key] = task
            def finished(done):
                if self._review_tasks.get(key) is done:
                    self._review_tasks.pop(key, None)
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(finished)
        return await asyncio.shield(task)

    def _review_unavailable(self, kind):
        symbols = self._review_symbols()
        schema_key = "review_schema_version" if kind == "contract" else "reply_review_schema_version"
        version = symbols.REVIEW_SCHEMA_VERSION if kind == "contract" else symbols.REPLY_REVIEW_SCHEMA_VERSION
        return {schema_key: version, "state": "unavailable", "reason": "复盘请求已失效，请重新刷新。",
                "provider_id": "", "model": "", "generated_at": None, "review": None, "stats": {}}

    async def contract_review_payload(self, *, refresh=False):
        return await self._review_singleflight("contract", refresh=refresh)

    async def reply_review_payload(self, *, refresh=False, sp_module=None):
        return await self._review_singleflight("reply", refresh=refresh, sp_module=sp_module)

    async def _contract_review_payload_impl(self, *, refresh: bool = False) -> dict[str, Any]:
        """The contract panel, reread by a model.

        The deterministic matrix is the input, not the answer: what a reader
        opens is the model's reading of it, with every number checked back
        against the digest it was given. Every failure — the feature is off, the
        host has no provider, the call times out, the reply is prose — returns
        the same shape with an empty review and the reason, because the page has
        a complete fallback table and a blank panel would be a worse answer than
        an uninterpreted one.
        """
        epoch = self._review_epoch
        quality = await self.quality_payload()
        digest = self._review_symbols().build_digest(quality)
        fingerprint = self._review_symbols().digest_fingerprint(digest)
        config = self.runtime_config()
        payload: dict[str, Any] = {
            "review_schema_version": self._review_symbols().REVIEW_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "state": "unavailable",
            "reason": "",
            "provider_id": "",
            "generated_at": None,
            "review": None,
        }
        if not config.review_enabled:
            payload["state"] = "disabled"
            payload["reason"] = ("模型解读已关闭（learning_review_enabled=false）；"
                                 "下面是本插件自己的判定与计数。")
            return payload

        provider_id, model = await self._review_provider(config.review_provider_id)
        fingerprint = self._review_identity("contract", fingerprint, provider_id, model, digest)
        payload["provider_id"] = provider_id
        payload["model"] = model
        payload["fingerprint"] = fingerprint
        if not provider_id:
            return dict(payload, state="unavailable", reason="宿主没有可用的对话模型 Provider，无法生成复盘。")
        cached = await self.store.load_review()
        if (not refresh and cached.get("fingerprint") == fingerprint
                and isinstance(cached.get("review"), dict)):
            if not await self._review_current(epoch, "review", provider_id, model):
                return self._review_unavailable("contract")
            payload.update(state="cached", generated_at=cached.get("generated_at"),
                           provider_id=str(cached.get("provider_id") or ""),
                           review=dict(cached["review"]))
            return payload

        # A provider that is down must not be called once per page refresh.
        if not refresh and self._review_failed_at and getattr(self, "_review_failure_key", None) == fingerprint:
            waited = time.time() - self._review_failed_at
            if waited < self._review_symbols().REVIEW_RETRY_SECONDS:
                payload["state"] = "failed"
                payload["reason"] = self._review_failure_reason
                return payload

        if not provider_id:
            payload["state"] = "unavailable"
            payload["reason"] = "宿主没有可用的对话模型 Provider，无法生成模型解读。"
            return payload
        payload["provider_id"] = provider_id
        payload["model"] = model

        try:
            reply = await asyncio.wait_for(
                self._ask_review(digest, provider_id),
                timeout=float(config.review_timeout_seconds))
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return self._review_failed(
                payload, f"模型在 {config.review_timeout_seconds} 秒内没有返回，已回落到本插件判定。")
        except Exception as exc:
            return self._review_failed(
                payload, f"调用模型失败（{type(exc).__name__}），已回落到本插件判定。")

        review = self._review_symbols().parse_review(reply, digest)
        if review is None:
            return self._review_failed(payload, "模型返回的不是可解析的 JSON 对象，已回落到本插件判定。")

        if not await self._review_current(epoch, "review", provider_id, model):
            return dict(payload, state="unavailable", reason="复盘配置已变化，请重新刷新。")
        self._review_failed_at = 0.0
        self._review_failure_reason = ""
        generated_at = time.time()
        review["provider_id"] = provider_id
        review["model"] = model
        saved = await self._save_review_if_current(epoch, {
            "review_schema_version": self._review_symbols().REVIEW_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "generated_at": generated_at,
            "provider_id": provider_id,
            "model": model,
            "review": review,
        })
        if not saved:
            return self._review_unavailable("contract")
        payload.update(state="fresh", generated_at=generated_at, review=review)
        return payload

    def _review_failed(self, payload: dict[str, Any], reason: str) -> dict[str, Any]:
        self._review_failure_key = payload.get("fingerprint")
        self._review_failed_at = time.time()
        self._review_failure_reason = reason
        payload["state"] = "failed"
        payload["reason"] = reason
        return payload

    async def _review_provider(self, explicit: str = "") -> tuple[str, str]:
        """The provider to ask, and its model name when the host knows it.

        An explicit id wins: a review is a fixed analytical task, and letting the
        panel follow whichever model a conversation happens to use means two
        readers of the same page can get different tables with no way to tell
        why. With nothing configured, the host's current chat provider is used,
        because a plugin that cannot be configured still has to say something.
        """
        if explicit:
            getter = getattr(getattr(self, "context", None), "get_provider_by_id", None)
            try:
                provider = getter(explicit) if callable(getter) else None
                if inspect.isawaitable(provider):
                    provider = await provider
                meta = getattr(provider, "meta", None)
                info = meta() if callable(meta) else None
            except Exception:
                info = None
            return explicit, str(getattr(info, "model", "") or "")
        context = getattr(self, "context", None)
        getter = getattr(context, "get_using_provider", None)
        if not callable(getter):
            return "", ""
        try:
            provider = getter(None)
        except TypeError:
            provider = getter()
        except Exception:
            return "", ""
        if inspect.isawaitable(provider):
            provider = await provider
        if provider is None:
            return "", ""
        meta = getattr(provider, "meta", None)
        try:
            info = meta() if callable(meta) else None
        except Exception:
            info = None
        return str(getattr(info, "id", "") or ""), str(getattr(info, "model", "") or "")

    async def _ask_review(self, digest: dict[str, Any], provider_id: str) -> str:
        context = getattr(self, "context", None)
        generate = getattr(context, "llm_generate", None)
        if not callable(generate):
            raise RuntimeError("AstrBot context does not expose llm_generate")
        system_prompt, prompt = self._review_symbols().build_prompt(digest)
        response = generate(chat_provider_id=provider_id, prompt=prompt, system_prompt=system_prompt)
        if inspect.isawaitable(response):
            response = await response
        return self._review_symbols()._completion_text(response)

    async def _reply_review_payload_impl(self, *, refresh: bool = False,
                                   sp_module: Any = None) -> dict[str, Any]:
        """Per-message post-mortem of the reply decision, written by a model.

        This is the one path in the plugin that reads message text. It is read
        from the host shared preferences for this call only, sent to the
        configured model, and never written back: not into the sample store,
        not into a cache. The result is held in memory for the life of the
        process, so a restart forgets it — which is also why the cache below
        is a plain dict and not a KV key.

        The model is not shown the human label or what the host decided. That
        is the whole point: a judge that has been shown the answer agrees with
        it, and the useful rows here are the ones where the three disagree.
        """
        epoch = self._review_epoch
        config = self.runtime_config()
        payload: dict[str, Any] = {
            "reply_review_schema_version": self._review_symbols().REPLY_REVIEW_SCHEMA_VERSION,
            "state": "unavailable",
            "reason": "",
            "provider_id": "",
            "model": "",
            "generated_at": None,
            "stats": {},
            "review": None,
            "text_policy": ("正文只在本体与模型之间过一次：本插件不保存正文，复盘结果也不落盘；"
                            "关掉开关后连读都不读。"),
        }
        if not config.reply_review_enabled:
            payload["state"] = "disabled"
            payload["reason"] = ("逐条复盘默认关闭：它会把群消息正文发给你配置的模型。"
                                 "确认接受这一点后，打开 learning_reply_review_enabled。")
            return payload

        try:
            result = await self._review_symbols().collect_from_host(config.source_plugin_id, sp_module=sp_module)
        except Exception as exc:
            payload["reason"] = f"读取本体标注失败（{type(exc).__name__}）。"
            return payload
        if result.diagnostics.get("available") is False:
            # self._review_symbols().collect_from_host reports an unreachable host instead of raising:
            # "the host is not there" and "the host has no annotations" are
            # different findings with different fixes.
            payload["reason"] = ("读不到本体的共享首选项（本体未安装、未加载，或 AstrBot 版本不支持）。")
            return payload
        messages, stats = self._review_symbols().select_messages(result.annotations,
                                          limit=config.reply_review_max_messages)
        payload["stats"] = stats
        if not messages:
            payload["state"] = "empty"
            payload["reason"] = ("本体还没有可复盘的标注记录：先在 ChatDynamics 的场景回放里标注"
                                 "「该不该回」（expected_reply）。")
            return payload
        if not stats["with_text"]:
            payload["state"] = "no_text"
            payload["reason"] = (
                f"选中的 {stats['selected']} 条都没有正文。本插件从不保存正文，本体也只在打开"
                "「控制台显示消息正文」时才把它写进标注记录；打开它并重新标注后即可复盘。")
            return payload

        digest = self._review_symbols().build_reply_digest(messages)
        fingerprint = self._review_symbols().reply_digest_fingerprint(digest)
        provider_id, model = await self._review_provider(config.reply_review_provider_id)
        fingerprint = self._review_identity("reply", fingerprint, provider_id, model, digest, messages)
        payload["provider_id"] = provider_id
        payload["model"] = model
        payload["fingerprint"] = fingerprint
        if not provider_id:
            return dict(payload, state="unavailable", reason="宿主没有可用的对话模型 Provider，无法生成复盘。")
        cached = self._reply_review_cache
        if not refresh and cached.get("fingerprint") == fingerprint:
            if not await self._review_current(epoch, "reply_review", provider_id, model):
                return self._review_unavailable("reply")
            payload.update(state="cached", generated_at=cached.get("generated_at"),
                           provider_id=cached.get("provider_id", ""),
                           model=cached.get("model", ""), review=cached.get("review"))
            return payload
        if not refresh and self._reply_review_failed_at and getattr(self, "_reply_review_failure_key", None) == fingerprint:
            if time.time() - self._reply_review_failed_at < self._review_symbols().REVIEW_RETRY_SECONDS:
                payload["state"] = "failed"
                payload["reason"] = self._reply_review_failure_reason
                return payload

        if not provider_id:
            payload["reason"] = "宿主没有可用的对话模型 Provider，无法复盘。"
            return payload
        payload["provider_id"] = provider_id
        payload["model"] = model
        try:
            reply = await asyncio.wait_for(
                self._ask_reply_review(digest, provider_id),
                timeout=float(config.reply_review_timeout_seconds))
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return self._reply_review_failed(
                payload, f"模型在 {config.reply_review_timeout_seconds} 秒内没有返回。")
        except Exception as exc:
            return self._reply_review_failed(
                payload, f"调用模型失败（{type(exc).__name__}）。")

        review = self._review_symbols().parse_reply_review(reply, messages, stats=stats)
        if review is None:
            return self._reply_review_failed(payload, "模型返回的不是可解析的 JSON 对象。")

        if not await self._review_current(epoch, "reply_review", provider_id, model):
            return dict(payload, state="unavailable", reason="复盘配置已变化，请重新刷新。")
        self._reply_review_failed_at = 0.0
        self._reply_review_failure_reason = ""
        generated_at = time.time()
        review["provider_id"] = provider_id
        review["model"] = model
        review["generated_at"] = generated_at
        self._reply_review_cache = {"fingerprint": fingerprint, "generated_at": generated_at,
                                    "provider_id": provider_id, "model": model,
                                    "review": review}
        payload.update(state="fresh", generated_at=generated_at, review=review)
        return payload

    def _reply_review_failed(self, payload: dict[str, Any], reason: str) -> dict[str, Any]:
        self._reply_review_failure_key = payload.get("fingerprint")
        self._reply_review_failed_at = time.time()
        self._reply_review_failure_reason = reason
        payload["state"] = "failed"
        payload["reason"] = reason
        return payload

    async def _ask_reply_review(self, digest: dict[str, Any], provider_id: str) -> str:
        context = getattr(self, "context", None)
        generate = getattr(context, "llm_generate", None)
        if not callable(generate):
            raise RuntimeError("AstrBot context does not expose llm_generate")
        system_prompt, prompt = self._review_symbols().build_reply_prompt(digest)
        response = generate(chat_provider_id=provider_id, prompt=prompt, system_prompt=system_prompt)
        if inspect.isawaitable(response):
            response = await response
        return self._review_symbols()._completion_text(response)
