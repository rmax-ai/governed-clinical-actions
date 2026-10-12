"""Durable workflow failure, audit, and restart-boundary tests."""

from __future__ import annotations

import pytest

from governed_clinical_actions.audit import AppendOnlyAuditLog
from governed_clinical_actions.models import (
    ExecutionPlan,
    PurposeOfUse,
    ResearchRequest,
    RunStatus,
    ToolEventStatus,
)
from governed_clinical_actions.provenance import request_id_for
from governed_clinical_actions.workflow import (
    IdempotencyKey,
    InMemoryStateStore,
    JsonStateStore,
    PersistedWorkflowState,
    RetryPolicy,
    ToolDefinition,
    ToolExecutionError,
    ToolRegistry,
    WorkflowEngine,
    WorkflowPhase,
)


def _identity(tool_names: tuple[str, ...]) -> tuple[ResearchRequest, ExecutionPlan]:
    draft = ResearchRequest(
        request_id="placeholder",
        original_intent="synthetic durability fixture",
        purpose_of_use=PurposeOfUse(code="commercial_ai_model_development"),
    )
    request = draft.model_copy(update={"request_id": request_id_for(draft)})
    plan = ExecutionPlan(
        plan_id="plan-durability",
        request_id=request.request_id,
        tool_names=list(tool_names),
    )
    return request, plan


def _engine(
    store: InMemoryStateStore | JsonStateStore,
    registry: ToolRegistry,
    tool_names: tuple[str, ...],
) -> WorkflowEngine:
    request, plan = _identity(tool_names)
    return WorkflowEngine(request, plan, registry, state_store=store)


def test_tool_failure_appends_tool_failed_and_persists_failed_state_before_propagation() -> None:
    calls = {"first": 0, "second": 0}

    def first_handler(arguments: object) -> dict[str, str]:
        calls["first"] += 1
        return {"step": "first"}

    def second_handler(arguments: object) -> None:
        calls["second"] += 1
        raise RuntimeError("synthetic second-step failure")

    store = InMemoryStateStore()
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="first", handler=first_handler))
    registry.register(ToolDefinition(name="second", handler=second_handler))
    engine = _engine(store, registry, ("first", "second"))

    with pytest.raises(ToolExecutionError) as raised:
        engine.run()

    persisted = store.load(engine.run_id)
    failed_events = [
        event
        for event in persisted.run.tool_events
        if event.status == ToolEventStatus.FAILED
    ]
    assert calls == {"first": 1, "second": 1}
    assert raised.value.error_code == "tool.handler_error"
    assert persisted.phase == WorkflowPhase.FAILED
    assert persisted.run.status == RunStatus.FAILED
    assert persisted.run.error_code == "tool.handler_error"
    assert any(event.event_type == "tool.failed" for event in persisted.audit_events)
    assert len(failed_events) == 1
    assert failed_events[0].error_code == "tool.handler_error"
    assert AppendOnlyAuditLog(persisted.audit_events).event_ids() == tuple(
        persisted.run.audit_event_ids
    )


def test_failed_state_and_audit_chain_survive_json_state_store_round_trip() -> None:
    calls = {"n": 0}

    def always_fails(arguments: object) -> None:
        calls["n"] += 1
        raise RuntimeError("synthetic retry exhaustion")

    store = JsonStateStore()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="retry-three",
            handler=always_fails,
            retry_policy=RetryPolicy(
                max_attempts=3,
                retryable_error_codes=frozenset({"tool.handler_error"}),
            ),
        )
    )
    engine = _engine(store, registry, ("retry-three",))

    with pytest.raises(ToolExecutionError):
        engine.run()

    serialized = store._payloads[engine.run_id]
    persisted = PersistedWorkflowState.model_validate_json_boundary(serialized)
    reloaded = store.load(engine.run_id)
    failed_event = next(
        event
        for event in persisted.run.tool_events
        if event.status == ToolEventStatus.FAILED
    )
    receipt = next(
        event
        for event in persisted.idempotency_receipts
        if event.idempotency_key == failed_event.idempotency_key
    )

    assert calls["n"] == 3
    assert persisted.phase == WorkflowPhase.FAILED
    assert persisted.run.status == RunStatus.FAILED
    assert persisted.run.error_code == "tool.handler_error"
    assert failed_event.attempt == 3
    assert failed_event.error_code == "tool.handler_error"
    assert receipt == failed_event
    assert reloaded == persisted
    assert AppendOnlyAuditLog(persisted.audit_events).event_ids() == tuple(
        persisted.run.audit_event_ids
    )

    legacy_payload = persisted.model_dump(mode="json")
    legacy_payload.pop("idempotency_receipts")
    legacy_state = PersistedWorkflowState.model_validate_boundary(legacy_payload)
    assert legacy_state.idempotency_receipts == []


def test_resume_of_a_failed_run_does_not_reexecute_the_failed_tool() -> None:
    calls = {"n": 0}

    def fails(arguments: object) -> None:
        calls["n"] += 1
        raise RuntimeError("synthetic failure")

    store = InMemoryStateStore()
    first_registry = ToolRegistry()
    first_registry.register(ToolDefinition(name="fails", handler=fails))
    first_engine = _engine(store, first_registry, ("fails",))
    with pytest.raises(ToolExecutionError):
        first_engine.run()
    assert calls["n"] == 1

    reconstructed_registry = ToolRegistry()
    reconstructed_registry.register(ToolDefinition(name="fails", handler=fails))
    reconstructed_engine = _engine(store, reconstructed_registry, ("fails",))
    result = reconstructed_engine.resume()

    assert result.phase == WorkflowPhase.FAILED
    assert calls["n"] == 1


def test_paused_resume_does_not_reexecute_a_completed_side_effect_after_reconstruction() -> None:
    calls = {"first": 0, "second": 0}

    def first_side_effect(arguments: object) -> dict[str, str]:
        calls["first"] += 1
        return {"completed": "first"}

    def second_step(arguments: object) -> dict[str, str]:
        calls["second"] += 1
        return {"completed": "second"}

    store = InMemoryStateStore()
    first_registry = ToolRegistry()
    first_registry.register(ToolDefinition(name="first", handler=first_side_effect))
    first_registry.register(ToolDefinition(name="second", handler=second_step))
    first_engine = _engine(store, first_registry, ("first", "second"))
    paused = first_engine.run(max_steps=1)

    assert paused.phase == WorkflowPhase.PAUSED
    assert calls == {"first": 1, "second": 0}

    reconstructed_registry = ToolRegistry()
    reconstructed_registry.register(ToolDefinition(name="first", handler=first_side_effect))
    reconstructed_registry.register(ToolDefinition(name="second", handler=second_step))
    reconstructed_engine = _engine(store, reconstructed_registry, ("first", "second"))
    completed = reconstructed_engine.resume()

    assert completed.phase == WorkflowPhase.COMPLETED
    assert calls == {"first": 1, "second": 1}


def test_write_ahead_receipt_is_persisted_before_the_handler_runs() -> None:
    store = InMemoryStateStore()
    observations: list[tuple[str, ToolEventStatus]] = []
    engine_holder: dict[str, WorkflowEngine] = {}

    def inspect_receipt(arguments: object) -> dict[str, bool]:
        engine = engine_holder["engine"]
        key = IdempotencyKey.for_call(
            engine.run_id,
            "observe",
            {"request_id": engine.request.request_id},
        )
        persisted = store.load(engine.run_id)
        receipt = next(
            event
            for event in persisted.idempotency_receipts
            if event.idempotency_key == key.value
        )
        observations.append((receipt.idempotency_key, receipt.status))
        return {"observed": True}

    registry = ToolRegistry()
    registry.register(ToolDefinition(name="observe", handler=inspect_receipt))
    engine = _engine(store, registry, ("observe",))
    engine_holder["engine"] = engine

    result = engine.run()

    expected_key = IdempotencyKey.for_call(
        engine.run_id,
        "observe",
        {"request_id": engine.request.request_id},
    )
    assert result.phase == WorkflowPhase.COMPLETED
    assert observations == [(expected_key.value, ToolEventStatus.STARTED)]


def test_missing_tool_failure_persists_terminal_failed_receipt_and_state() -> None:
    store = InMemoryStateStore()
    registry = ToolRegistry()
    engine = _engine(store, registry, ("ghost",))

    with pytest.raises(ToolExecutionError) as raised:
        engine.run()

    state = store.load(engine.run_id)
    expected_key = IdempotencyKey.for_call(
        engine.run_id,
        "ghost",
        {"request_id": engine.request.request_id},
    )
    failed_audit_events = [
        event for event in state.audit_events if event.event_type == "tool.failed"
    ]
    failed_tool_events = [
        event
        for event in state.run.tool_events
        if event.status == ToolEventStatus.FAILED
    ]
    receipts_for_key = [
        event
        for event in state.idempotency_receipts
        if event.idempotency_key == expected_key.value
    ]

    assert raised.value.error_code == "tool.not_found"
    assert state.phase == WorkflowPhase.FAILED
    assert state.run.status == RunStatus.FAILED
    assert state.run.error_code == "tool.not_found"
    assert len(failed_audit_events) == 1
    assert len(failed_tool_events) == 1
    assert failed_tool_events[0].error_code == "tool.not_found"
    assert len(receipts_for_key) == 1
    assert receipts_for_key[0].status == ToolEventStatus.FAILED
    assert receipts_for_key[0].error_code == "tool.not_found"
    assert AppendOnlyAuditLog(state.audit_events).event_ids() == tuple(
        state.run.audit_event_ids
    )
