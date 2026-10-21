# Copyright (c) 2026 LangChain, Inc.
"""Cancellation drains both loops before destroying an interpreter slot."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs._repl import _Registry
from tests._common import FakeChatModel


class _InvocationRecorder(AgentMiddleware):
    invocation: tuple[Any, Any] | None = None

    async def abefore_model(self, state: Any, runtime: Any) -> None:
        self.invocation = state, runtime


@pytest.mark.parametrize("cancel_eval", [False, True])
@pytest.mark.parametrize("cancel_close", [False, True])
async def test_close_waits_for_ptc_cleanup_before_context_destruction(
    cancel_eval: bool,
    cancel_close: bool,
) -> None:
    entered, cleaning, release, finished = (asyncio.Event() for _ in range(4))
    errors: list[dict[str, Any]] = []

    @tool
    async def slow_read() -> int:
        """Hold the native PTC call until cancellation is drained."""
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()
        return 42

    registry = _Registry(
        memory_limit=64 * 1024 * 1024,
        timeout=10,
        capture_console=True,
        max_stdout_chars=4096,
    )
    repl = registry.get("cancellation-cleanup")
    worker = registry._slots["cancellation-cleanup"].worker
    baseline = {thread.ident for thread in threading.enumerate()}

    async def capture_errors() -> None:
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, data: errors.append(data)
        )

    await worker.run_async(capture_errors())
    repl.install_tools([slow_read])
    evaluation = asyncio.create_task(
        repl.eval_async(
            "await tools.slowRead({})",
            outer_loop=asyncio.get_running_loop(),
        )
    )
    closing: asyncio.Task[None] | None = None
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            if cancel_eval:
                evaluation.cancel()
                await cleaning.wait()
                assert not evaluation.done()
            closing = asyncio.create_task(registry.aevict("cancellation-cleanup"))
            await cleaning.wait()
            # A worker-loop round trip proves the close request has been serviced.
            await worker.run_async(asyncio.sleep(0))
            await _finish_cancelled_work(
                evaluation, closing, release, finished, cancel_close=cancel_close
            )
        assert finished.is_set()
        assert errors == []
        assert _worker_stopped(baseline)
    finally:
        release.set()
        await asyncio.gather(evaluation, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await registry.aevict("cancellation-cleanup")


def _worker_stopped(baseline: set[int | None]) -> bool:
    return not any(
        thread.ident in baseline and thread.name.startswith("quickjs-worker-cancella")
        for thread in threading.enumerate()
    )


async def _finish_cancelled_work(
    evaluation: asyncio.Task[Any],
    closing: asyncio.Task[None],
    release: asyncio.Event,
    finished: asyncio.Event,
    *,
    cancel_close: bool,
) -> None:
    if cancel_close:
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
    assert not closing.done()
    assert not evaluation.done()
    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await evaluation
    if cancel_close:
        with pytest.raises(asyncio.CancelledError):
            await closing
    else:
        await closing


async def test_cancel_fanout_drains_active_workers_and_never_starts_queued_work() -> (
    None
):
    entered, cleaning, release = (asyncio.Event() for _ in range(3))
    started = active = finished = 0
    errors: list[dict[str, Any]] = []

    async def work(_state: Any) -> dict[str, Any]:
        nonlocal started, active, finished
        started += 1
        active += 1
        if active == 2:
            entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaning.set()
            await release.wait()
            active -= 1
            finished += 1
        return {"messages": [AIMessage(content="unused")]}

    interpreter = CodeInterpreterMiddleware(
        mode="turn",
        subagents=True,
        max_subagent_concurrency=2,
    )
    recorder = _InvocationRecorder()
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "eval",
                "id": "fanout",
                "args": {
                    "code": (
                        "await Promise.all(Array.from({length:7},(_,i)=>task({"
                        "description:'Hold '+i,subagentType:'held-worker'})))"
                    )
                },
            }
        ],
    )
    graph = create_deep_agent(
        model=FakeChatModel(messages=iter([message])),
        middleware=[interpreter, recorder],
        subagents=[
            {
                "name": "held-worker",
                "description": "Wait for cancellation",
                "runnable": RunnableLambda(work),
            }
        ],
    )
    invocation = asyncio.create_task(
        graph.ainvoke({"messages": [HumanMessage(content="Fan out")]})
    )
    closing: asyncio.Task[Any] | None = None
    try:
        async with asyncio.timeout(10):
            await entered.wait()
            await _capture_interpreter_errors(interpreter, errors)
            invocation.cancel()
            await cleaning.wait()
            assert recorder.invocation is not None
            closing = asyncio.create_task(
                interpreter.aafter_agent(*recorder.invocation)
            )
            # The callback is serviced, but held worker cleanup cannot finish yet.
            await asyncio.sleep(0)
            assert not closing.done()
            assert started == 2
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await invocation
            await closing
        assert active == 0
        assert finished == started == 2
        assert errors == []
    finally:
        release.set()
        await asyncio.gather(invocation, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        if recorder.invocation is not None:
            await interpreter.aafter_agent(*recorder.invocation)


async def _capture_interpreter_errors(
    interpreter: CodeInterpreterMiddleware, errors: list[dict[str, Any]]
) -> None:
    async def capture_errors() -> None:
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, data: errors.append(data)
        )

    for slot in interpreter._registry._slots.values():
        await slot.worker.run_async(capture_errors())
