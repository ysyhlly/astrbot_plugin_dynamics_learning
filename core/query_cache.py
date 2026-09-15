"""Bounded query cache with generation-safe asynchronous computation."""
from __future__ import annotations

import asyncio
import copy
import time
from collections import OrderedDict
from typing import Any, Callable


class QueryCache:
    def __init__(self, limit: int = 32, ttl: float = 30.0):
        self.generation = 0
        self.limit = limit
        self.ttl = ttl
        self.values: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self.pending: dict[Any, asyncio.Task] = {}

    def invalidate(self) -> None:
        self.generation += 1
        self.values.clear()

    async def compute(self, key: Any, function: Callable, *args: Any, **kwargs: Any) -> Any:
        identity = (self.generation, key)
        cached = self.values.get(identity)
        if cached is not None and time.monotonic() - cached[0] < self.ttl:
            self.values.move_to_end(identity)
            return copy.deepcopy(cached[1])
        task = self.pending.get(identity)
        if task is None:
            async def run():
                try:
                    result = await asyncio.to_thread(function, *args, **kwargs)
                    if identity[0] == self.generation:
                        self.values[identity] = (time.monotonic(), copy.deepcopy(result))
                        while len(self.values) > self.limit:
                            self.values.popitem(last=False)
                    return result
                finally:
                    self.pending.pop(identity, None)
            task = asyncio.create_task(run())
            self.pending[identity] = task
        return copy.deepcopy(await asyncio.shield(task))

    async def close(self) -> None:
        self.invalidate()
        tasks = list(self.pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
