"""Tool failure codes, retry semantics, and durable idempotency receipts."""

from __future__ import annotations

import pytest

from governed_clinical_actions.models import ToolEvent, ToolEventStatus, stable_digest
from governed_clinical_actions.workflow import (
    IdempotencyKey,
    IdempotencyReceiptLedger,
    RetryPolicy,
    ToolCall,
    ToolDefinition,
    ToolExecutionError,
    ToolGate,
    ToolRegistry,
)


def _execute(gate: ToolGate, tool_name: str, run_id: str = "run-durable") -> None:
    gate.execute(
        run_id=run_id,
        call=ToolCall(name=tool_name, arguments={"input": "synthetic"}),
        sequence=1,
    )


def test_handler_failure_carries_failed_event_with_stable_error_code() -> None:
    calls = {"n": 0}

    def handler(arguments: object) -> None:
        calls["n"] += 1
        raise RuntimeError("synthetic handler failure")

    registry = ToolRegistry()
    registry.register(ToolDefinition(name="fails", handler=handler))
    receipts = IdempotencyReceiptLedger()

    with pytest.raises(ToolExecutionError) as raised:
        _execute(ToolGate(registry, ("fails",), receipts=receipts), "fails")

    event = raised.value.event
    assert calls["n"] == 1
    assert raised.value.error_code == "tool.handler_error"
    assert event is not None
    assert event.status == ToolEventStatus.FAILED
    assert event.error_code == "tool.handler_error"
    assert receipts.get(event.idempotency_key) == event


def test_output_validation_failure_carries_failed_event_with_stable_error_code() -> None:
    calls = {"n": 0}

    def handler(arguments: object) -> str:
        calls["n"] += 1
        return "not-an-integer"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="invalid-output", handler=handler, output_type=int)
    )

    with pytest.raises(ToolExecutionError) as raised:
        _execute(ToolGate(registry, ("invalid-output",)), "invalid-output")

    event = raised.value.event
    assert calls["n"] == 1
    assert raised.value.error_code == "tool.output_invalid"
    assert event is not None
    assert event.status == ToolEventStatus.FAILED
    assert event.error_code == "tool.output_invalid"


def test_exhausted_retry_raises_once_after_max_attempts_with_single_failed_event() -> None:
    calls = {"n": 0}

    def handler(arguments: object) -> None:
        calls["n"] += 1
        raise RuntimeError("synthetic retryable failure")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="retry-three",
            handler=handler,
            retry_policy=RetryPolicy(
                max_attempts=3,
                retryable_error_codes=frozenset({"tool.handler_error"}),
            ),
        )
    )
    receipts = IdempotencyReceiptLedger()

    with pytest.raises(ToolExecutionError) as raised:
        _execute(ToolGate(registry, ("retry-three",), receipts=receipts), "retry-three")

    event = raised.value.event
    assert calls["n"] == 3
    assert event is not None
    assert event.attempt == 3
    assert event.status == ToolEventStatus.FAILED
    assert event.error_code == "tool.handler_error"
    assert raised.value.error_code == "tool.handler_error"
    key_receipts = [
        receipt
        for receipt in receipts.as_events()
        if receipt.idempotency_key == event.idempotency_key
    ]
    assert key_receipts == [event]


def test_retry_policy_matches_stable_error_code_not_tool_name() -> None:
    name_match_calls = {"n": 0}
    code_match_calls = {"n": 0}
    retry_policy = RetryPolicy(
        max_attempts=3,
        retryable_error_codes=frozenset({"tool.handler_error"}),
    )

    def name_matches_code(arguments: object) -> None:
        name_match_calls["n"] += 1
        raise ToolExecutionError(
            "tool.handler_error",
            "synthetic different failure",
            error_code="tool.output_invalid",
        )

    def stable_code_matches(arguments: object) -> None:
        code_match_calls["n"] += 1
        raise RuntimeError("synthetic handler failure")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="tool.handler_error",
            handler=name_matches_code,
            retry_policy=retry_policy,
        )
    )
    registry.register(
        ToolDefinition(
            name="other-tool",
            handler=stable_code_matches,
            retry_policy=retry_policy,
        )
    )
    gate = ToolGate(registry, ("tool.handler_error", "other-tool"))

    with pytest.raises(ToolExecutionError):
        _execute(gate, "tool.handler_error")
    with pytest.raises(ToolExecutionError):
        _execute(gate, "other-tool")

    assert name_match_calls["n"] == 1
    assert code_match_calls["n"] == 3


def test_reconstructed_registry_does_not_reexecute_a_completed_side_effect() -> None:
    calls = {"n": 0}

    def side_effect(arguments: object) -> dict[str, bool]:
        calls["n"] += 1
        return {"completed": True}

    original_registry = ToolRegistry()
    original_registry.register(ToolDefinition(name="side-effect", handler=side_effect))
    original_receipts = IdempotencyReceiptLedger()
    original_gate = ToolGate(
        original_registry,
        ("side-effect",),
        receipts=original_receipts,
    )
    call = ToolCall(name="side-effect", arguments={"input": "synthetic"})
    first = original_gate.execute(run_id="run-reconstructed", call=call, sequence=1)
    assert first.event.status == ToolEventStatus.SUCCEEDED
    assert calls["n"] == 1

    reconstructed_registry = ToolRegistry()
    reconstructed_registry.register(ToolDefinition(name="side-effect", handler=side_effect))
    reconstructed_receipts = IdempotencyReceiptLedger(original_receipts.as_events())
    reconstructed_gate = ToolGate(
        reconstructed_registry,
        ("side-effect",),
        receipts=reconstructed_receipts,
    )
    with pytest.raises(ToolExecutionError) as raised:
        reconstructed_gate.execute(
            run_id="run-reconstructed",
            call=call,
            sequence=1,
        )

    assert raised.value.error_code == "tool.idempotency_replay_unavailable"
    assert calls["n"] == 1


def test_reconstructed_registry_refuses_ambiguous_started_receipt_without_executing() -> None:
    calls = {"n": 0}
    run_id = "run-ambiguous"
    tool_name = "side-effect"
    arguments = {"input": "synthetic"}
    key = IdempotencyKey.for_call(run_id, tool_name, arguments)
    started = ToolEvent(
        run_id=run_id,
        sequence=1,
        tool_name=tool_name,
        idempotency_key=key.value,
        status=ToolEventStatus.STARTED,
        input_digest=stable_digest(arguments),
        attempt=1,
    )

    def side_effect(value: object) -> dict[str, bool]:
        calls["n"] += 1
        return {"completed": True}

    registry = ToolRegistry()
    registry.register(ToolDefinition(name=tool_name, handler=side_effect))
    gate = ToolGate(
        registry,
        (tool_name,),
        receipts=IdempotencyReceiptLedger((started,)),
    )

    with pytest.raises(ToolExecutionError) as raised:
        gate.execute(
            run_id=run_id,
            call=ToolCall(name=tool_name, arguments=arguments),
            sequence=1,
        )

    assert raised.value.error_code == "tool.idempotency_ambiguous"
    assert calls["n"] == 0
