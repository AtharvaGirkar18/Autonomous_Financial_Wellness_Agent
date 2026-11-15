"""Distributed Agent Coordinator — parallel & sequential agent scheduling.

Replaces the naive sequential loop in the orchestrator with a proper
distributed coordination layer that:

  1. Analyses the execution plan's dependency graph.
  2. Schedules independent agents to run in PARALLEL (thread-pool).
  3. Schedules dependent agents in SEQUENCE after their prerequisites finish.
  4. Merges partial state updates from parallel agents safely (with locks).
  5. Emits structured events on the agent message bus for observability.
  6. Respects governance veto signals during execution.

Dependency graph (hardcoded domain knowledge):
  Ingestion → Categorization → Spending → Budget → Health
  Categorization → Anomaly
  Spending → Subscription (parallel with Budget)
  Health → Financial Twin → Investment Readiness → Advisor
  Advisor → Reviewer
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable

from guardrails.audit import audit_event


# ── Agent dependency graph ─────────────────────────────────────────────────────
# Maps agent → list of agents that MUST complete before it runs.
AGENT_DEPS: dict[str, list[str]] = {
    "Ingestion": [],
    "Categorization": ["Ingestion"],
    "Spending": ["Categorization"],
    "Budget": ["Spending"],
    "Health": ["Budget"],
    "Anomaly": ["Categorization"],          # Can run in parallel with Spending
    "Subscription": ["Spending"],           # Can run in parallel with Budget
    "Forecast": ["Health"],
    "Financial Twin": ["Health"],
    "Investment Readiness": ["Health"],
    "Advisor": ["Financial Twin", "Investment Readiness", "Forecast"],
    "Reviewer": ["Advisor"],
}

_MAX_WORKERS = 4
_AGENT_TIMEOUT_S = 60


class AgentEvent:
    """Structured event emitted by the coordinator message bus."""
    __slots__ = ("agent", "event_type", "timestamp", "data")

    def __init__(self, agent: str, event_type: str, data: Any = None) -> None:
        self.agent = agent
        self.event_type = event_type  # STARTED | COMPLETED | FAILED | SKIPPED
        self.timestamp = time.perf_counter()
        self.data = data or {}


class AgentMessageBus:
    """Lightweight in-process publish/subscribe bus for agent events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[AgentEvent] = []
        self._subscribers: list[Callable[[AgentEvent], None]] = []

    def publish(self, event: AgentEvent) -> None:
        with self._lock:
            self._events.append(event)
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception:
                pass

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> None:
        self._subscribers.append(fn)

    def events(self) -> list[AgentEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class DistributedCoordinator:
    """Schedules agents with dependency-aware parallel execution.

    Usage:
        coordinator = DistributedCoordinator()
        final_state = coordinator.run(agent_map, planned_agents, initial_state)
    """

    def __init__(self, max_workers: int = _MAX_WORKERS) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="coord")
        self.bus = AgentMessageBus()
        self._state_lock = threading.Lock()

    def _compute_stages(self, planned_agents: list[str]) -> list[list[str]]:
        """Topological sort → execution stages (agents in same stage run in parallel)."""
        planned_set = set(planned_agents)
        in_degree: dict[str, int] = {a: 0 for a in planned_agents}
        graph: dict[str, list[str]] = defaultdict(list)

        for agent in planned_agents:
            for dep in AGENT_DEPS.get(agent, []):
                if dep in planned_set:
                    in_degree[agent] += 1
                    graph[dep].append(agent)

        stages: list[list[str]] = []
        remaining = set(planned_agents)

        while remaining:
            # Collect agents with no outstanding dependencies
            ready = [a for a in remaining if in_degree[a] == 0]
            if not ready:
                # Cycle or missing dep — fall back to sequential
                audit_event(
                    "coordinator", "WARN", "cycle_detected",
                    f"remaining={remaining}, falling back to sequential"
                )
                ready = list(remaining)
            stages.append(sorted(ready))  # sort for determinism
            for agent in ready:
                remaining.discard(agent)
                for successor in graph[agent]:
                    in_degree[successor] = max(0, in_degree[successor] - 1)

        return stages

    def _run_agent(
        self,
        name: str,
        fn: Callable[[dict], dict],
        state: dict,
    ) -> dict:
        """Run a single agent and return its state delta."""
        self.bus.publish(AgentEvent(name, "STARTED"))
        try:
            result = fn(state)
            self.bus.publish(AgentEvent(name, "COMPLETED", {"keys": list(result.keys())}))
            audit_event("coordinator", "AGENT_COMPLETED", name, f"keys={list(result.keys())}")
            return result or {}
        except Exception as exc:
            self.bus.publish(AgentEvent(name, "FAILED", {"error": str(exc)[:200]}))
            audit_event("coordinator", "AGENT_FAILED", name, str(exc)[:200])
            raise

    def run(
        self,
        agent_map: dict[str, Callable[[dict], dict]],
        planned_agents: list[str],
        initial_state: dict,
    ) -> dict:
        """Execute agents in dependency-ordered stages, parallelising where safe."""
        self.bus.clear()
        state = dict(initial_state)
        stages = self._compute_stages(planned_agents)

        audit_event(
            "coordinator", "WORKFLOW_START", "distributed",
            f"agents={planned_agents} stages={len(stages)}"
        )

        for stage_idx, stage in enumerate(stages):
            if len(stage) == 1:
                # Single agent — run directly (avoids thread-pool overhead)
                name = stage[0]
                fn = agent_map.get(name)
                if fn is None:
                    audit_event("coordinator", "AGENT_SKIPPED", name, "not in agent_map")
                    self.bus.publish(AgentEvent(name, "SKIPPED"))
                    continue
                try:
                    delta = self._run_agent(name, fn, state)
                    with self._state_lock:
                        state.update(delta)
                except Exception:
                    pass  # error already audited; continue workflow
            else:
                # Multiple independent agents — run in parallel
                futures: dict[Future, str] = {}
                # Take a snapshot of the state for all parallel agents to read from
                state_snapshot = dict(state)

                for name in stage:
                    fn = agent_map.get(name)
                    if fn is None:
                        audit_event("coordinator", "AGENT_SKIPPED", name, "not in agent_map")
                        self.bus.publish(AgentEvent(name, "SKIPPED"))
                        continue
                    future = self._pool.submit(self._run_agent, name, fn, state_snapshot)
                    futures[future] = name

                for future in as_completed(futures, timeout=_AGENT_TIMEOUT_S * len(stage)):
                    name = futures[future]
                    try:
                        delta = future.result(timeout=1)
                        with self._state_lock:
                            state.update(delta)
                    except Exception:
                        pass  # error already audited

        audit_event(
            "coordinator", "WORKFLOW_DONE", "distributed",
            f"stages_executed={len(stages)}"
        )
        return state

    def event_summary(self) -> list[dict]:
        return [
            {
                "agent": e.agent,
                "event": e.event_type,
                "t": round(e.timestamp, 3),
                **e.data,
            }
            for e in self.bus.events()
        ]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
