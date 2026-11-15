"""MCP (Model Context Protocol) package.

Provides a dynamic registry of swappable tool backends so the planner can
discover and call tools at runtime without being hard-wired to fixed APIs.
"""
from mcp.registry import MCPRegistry, mcp_registry, get_tool, list_tools  # noqa: F401

