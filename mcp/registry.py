"""MCP Tool Registry — multiple swappable MCP backends.

The registry acts as a unified broker for all agent tools. Instead of hardcoding
tool imports in each agent, agents call `mcp_registry.invoke(tool_name, **kwargs)`.
New backends (REST APIs, local functions, external MCPs) can be registered without
changing agent code.

Design:
  - Each MCP backend registers a set of named tools with metadata.
  - The planner lists available tools per backend for intent routing.
  - Tools are called through the registry with full audit logging.
  - Backends can be enabled/disabled at runtime (governance hook).
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from guardrails.audit import audit_event


@dataclass
class ToolSpec:
    """Metadata for a registered MCP tool."""
    name: str
    description: str
    backend: str
    fn: Callable[..., Any]
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def invoke(self, **kwargs: Any) -> Any:
        if not self.enabled:
            raise RuntimeError(f"Tool '{self.name}' is disabled by governance policy.")
        return self.fn(**kwargs)


class MCPRegistry:
    """Central registry for all MCP tool backends.

    Usage:
        registry = MCPRegistry()
        registry.register_backend("finance_mcp", tools_dict)
        result = registry.invoke("calculate_budget_utilization", user_id="U1")
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._backends: dict[str, list[str]] = {}  # backend → tool names

    def register_backend(self, backend_name: str, tools: dict[str, dict]) -> None:
        """Register a backend with its tools.

        Args:
            backend_name: Unique backend identifier (e.g. "finance_mcp").
            tools: Dict of {tool_name: {"fn": callable, "description": str, "tags": [...]}}
        """
        registered: list[str] = []
        for tool_name, spec in tools.items():
            ts = ToolSpec(
                name=tool_name,
                description=spec.get("description", ""),
                backend=backend_name,
                fn=spec["fn"],
                tags=spec.get("tags", []),
                enabled=spec.get("enabled", True),
            )
            self._tools[tool_name] = ts
            registered.append(tool_name)
        self._backends[backend_name] = registered
        audit_event("mcp", "BACKEND_REGISTERED", backend_name, f"tools={registered}")

    def invoke(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a registered tool by name with full audit logging."""
        spec = self._tools.get(tool_name)
        if spec is None:
            audit_event("mcp", "TOOL_NOT_FOUND", tool_name, str(kwargs)[:120])
            raise KeyError(f"MCP tool '{tool_name}' not found. Available: {list(self._tools)}")
        started = time.perf_counter()
        try:
            # Only pass kwargs that the function actually accepts
            sig = inspect.signature(spec.fn)
            accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
            result = spec.invoke(**accepted)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            audit_event("mcp", "TOOL_OK", tool_name, f"backend={spec.backend} latency={elapsed_ms}ms")
            return result
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            audit_event("mcp", "TOOL_ERROR", tool_name, f"{type(exc).__name__}: {str(exc)[:200]}")
            raise

    def list_tools(self, backend: str | None = None, tag: str | None = None) -> list[dict]:
        """List all registered tools, optionally filtered by backend or tag."""
        results = []
        for name, spec in self._tools.items():
            if backend and spec.backend != backend:
                continue
            if tag and tag not in spec.tags:
                continue
            results.append({
                "name": name,
                "description": spec.description,
                "backend": spec.backend,
                "tags": spec.tags,
                "enabled": spec.enabled,
            })
        return results

    def enable_tool(self, tool_name: str, enabled: bool = True) -> None:
        """Governance hook: enable or disable a tool at runtime."""
        if tool_name in self._tools:
            self._tools[tool_name].enabled = enabled
            audit_event("mcp", "TOOL_TOGGLED", tool_name, f"enabled={enabled}")

    def enable_backend(self, backend_name: str, enabled: bool = True) -> None:
        """Governance hook: enable or disable an entire backend."""
        for tname in self._backends.get(backend_name, []):
            self.enable_tool(tname, enabled)

    def get_schema_for_planner(self) -> str:
        """Return a concise tool listing string for the LLM planner's system prompt."""
        lines = ["Available MCP tools (call via mcp_registry.invoke):"]
        for backend, tnames in self._backends.items():
            lines.append(f"\n[{backend}]")
            for tname in tnames:
                spec = self._tools[tname]
                status = "" if spec.enabled else " [DISABLED]"
                lines.append(f"  - {tname}{status}: {spec.description}")
        return "\n".join(lines)


# ── Singleton registry used by all agents ─────────────────────────────────────
mcp_registry = MCPRegistry()


def get_tool(name: str) -> ToolSpec | None:
    return mcp_registry._tools.get(name)


def list_tools(**kwargs: Any) -> list[dict]:
    return mcp_registry.list_tools(**kwargs)

