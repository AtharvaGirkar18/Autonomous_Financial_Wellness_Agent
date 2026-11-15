from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = ROOT_DIR / "financial_wellness.db"


@dataclass(frozen=True)
class Settings:
    # ── LLM Provider ──────────────────────────────────────────────────────────
    # Default: ollama (local Qwen 2.5 7B — no API key required)
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    model_name: str = os.getenv("MODEL_NAME", "qwen2.5:7b")

    # ── Ollama (local, no key needed) ─────────────────────────────────────────
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "ollama")

    # ── Cloud providers (optional fallback) ───────────────────────────────────
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    xai_api_key: str = os.getenv("XAI_API_KEY", "")

    # ── Database ──────────────────────────────────────────────────────────────
    sqlite_url: str = os.getenv("DATABASE_URL") or f"sqlite:///{DB_PATH}"

    # ── MCP ───────────────────────────────────────────────────────────────────
    # Comma-separated list of MCP backends to activate at startup.
    mcp_backends: str = os.getenv("MCP_BACKENDS", "finance_mcp,analytics_mcp,data_mcp")

    # ── Governance ────────────────────────────────────────────────────────────
    governance_enabled: bool = os.getenv("GOVERNANCE_ENABLED", "true").lower() == "true"
    coordinator_max_workers: int = int(os.getenv("COORDINATOR_MAX_WORKERS", "4"))


settings = Settings()
