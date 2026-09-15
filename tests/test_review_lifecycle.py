"""Review cache provenance and concurrent lifecycle regression tests."""
import asyncio
import json
from types import SimpleNamespace
import pytest


def provider(name):
    return SimpleNamespace(meta=lambda: SimpleNamespace(id=name, model=name + "-model"))


@pytest.mark.asyncio
async def test_provider_switch_invalidates_cache(plugin, fake_context):
    fake_context.provider = provider("a")
    fake_context.llm_response = json.dumps({"headline": "ok"})
    assert (await plugin.contract_review_payload())["state"] == "fresh"
    fake_context.provider = provider("b")
    second = await plugin.contract_review_payload()
    assert second["state"] == "fresh"
    assert second["provider_id"] == "b"
    assert len(fake_context.llm_calls) == 2


@pytest.mark.asyncio
async def test_parallel_refresh_is_singleflight(plugin, fake_context):
    fake_context.provider = provider("a")
    started, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def ask(*args):
        calls.append(args)
        started.set()
        await release.wait()
        return json.dumps({"headline": "ok"})
    plugin._ask_review = ask
    tasks = [asyncio.create_task(plugin.contract_review_payload(refresh=True)) for _ in range(5)]
    await started.wait()
    release.set()
    results = await asyncio.gather(*tasks)
    assert len(calls) == 1
    assert all(result["state"] == "fresh" for result in results)


@pytest.mark.asyncio
async def test_disabling_during_generation_does_not_save(plugin, fake_context):
    fake_context.provider = provider("a")
    started, release = asyncio.Event(), asyncio.Event()
    async def ask(*args):
        started.set()
        await release.wait()
        return json.dumps({"headline": "ok"})
    plugin._ask_review = ask
    task = asyncio.create_task(plugin.contract_review_payload())
    await started.wait()
    plugin.config = {"learning_review_enabled": False}
    release.set()
    result = await task
    assert result["review"] is None
    assert not await plugin.store.load_review()


@pytest.mark.asyncio
async def test_invalidation_cancels_generation_and_prevents_cache_fill(plugin, fake_context):
    fake_context.provider = provider("a")
    started = asyncio.Event()
    async def ask(*args):
        started.set()
        await asyncio.Event().wait()
    plugin._ask_review = ask
    task = asyncio.create_task(plugin.contract_review_payload())
    await started.wait()
    await plugin.invalidate_reviews()
    result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    assert not await plugin.store.load_review()


@pytest.mark.asyncio
async def test_reset_storage_cancels_old_review(plugin, fake_context):
    fake_context.provider = provider("a")
    started = asyncio.Event()
    async def ask(*args):
        started.set()
        await asyncio.Event().wait()
    plugin._ask_review = ask
    task = asyncio.create_task(plugin.contract_review_payload())
    await started.wait()
    await plugin.reset_storage()
    result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    assert not await plugin.store.load_review()


@pytest.mark.asyncio
async def test_terminated_plugin_rejects_new_review(plugin, fake_context):
    fake_context.provider = provider("a")
    await plugin.terminate()
    assert (await plugin.contract_review_payload())["state"] == "unavailable"
    assert (await plugin.reply_review_payload())["state"] == "unavailable"
    assert not fake_context.llm_calls
    assert not plugin._review_tasks


@pytest.mark.asyncio
async def test_singleflight_separates_changed_model_and_reply_source(plugin, fake_context):
    plugin.config = {"learning_reply_review_enabled": True}
    started = asyncio.Queue()
    release = asyncio.Event()
    async def impl(**kwargs):
        await started.put(True)
        await release.wait()
        return {"state": "fresh"}
    plugin._reply_review_payload_impl = impl
    fake_context.provider = provider("a")
    source = object()
    first = asyncio.create_task(plugin.reply_review_payload(sp_module=source))
    await started.get()
    fake_context.provider = SimpleNamespace(meta=lambda: SimpleNamespace(id="a", model="changed"))
    second = asyncio.create_task(plugin.reply_review_payload(sp_module=source))
    await started.get()
    third = asyncio.create_task(plugin.reply_review_payload(sp_module=object()))
    await asyncio.sleep(0)
    assert len(plugin._review_tasks) == 3
    release.set()
    await asyncio.gather(first, second, third)
