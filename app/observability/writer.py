from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.observability.events import ObservabilityEvent
from app.observability.state.memory import InMemoryObservabilityState
from app.observability.storage.ring import RingEventStorage


@dataclass
class _WriteRequest:
    event: ObservabilityEvent
    done: asyncio.Future[None]


class ObservabilityWriter:
    def __init__(
        self,
        storage: RingEventStorage,
        state: InMemoryObservabilityState,
        *,
        queue_size: int = 10000,
    ) -> None:
        self._storage = storage
        self._state = state
        self._queue: asyncio.Queue[_WriteRequest | None] = asyncio.Queue(
            maxsize=queue_size
        )
        self._task: asyncio.Task[None] | None = None
        self._stopped = False

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopped = False
            self._task = asyncio.create_task(self._run(), name="observability-writer")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopped = True
        await self._queue.put(None)
        await self._task
        self._task = None

    async def write(self, event: ObservabilityEvent) -> ObservabilityEvent:
        if self._task is None or self._task.done() or self._stopped:
            raise RuntimeError("Observability writer is not running")
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._queue.put(_WriteRequest(event, done))
        await done
        return event

    def replay_state(self, limit: int | None = None) -> None:
        for event in self._storage.replay(limit=limit):
            self._state.apply(event)

    async def _run(self) -> None:
        while True:
            request = await self._queue.get()
            if request is None:
                self._queue.task_done()
                return
            try:
                await asyncio.to_thread(self._storage.append, request.event)
                self._state.apply(request.event)
            except Exception as exc:
                request.done.set_exception(exc)
            else:
                request.done.set_result(None)
            finally:
                self._queue.task_done()
