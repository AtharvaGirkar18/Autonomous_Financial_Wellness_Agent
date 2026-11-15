"""GuardrailBus — unified, composable guardrail pipeline.

The GuardrailBus is a single choke-point that all requests and responses
pass through. It chains multiple guardrail layers in a deterministic order
and returns a structured verdict. New layers can be added as plugins without
touching agent or orchestrator code.

Layers (in order):
  INPUT  → structural → PII redaction → rate limit → attack patterns → domain policy
  OUTPUT → secrets leak → securities pick → numeric grounding → PII → truncation

The bus replaces the ad-hoc per-agent guardrail calls with a single coherent
governance surface that is audited, observable, and configurable at runtime.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from guardrails.audit import audit_event
from guardrails.input_guard import GuardResult, check_query
from guardrails.output_guard import OutputCheck, check_output
from guardrails.pii import redact_pii
from guardrails.rate_limit import RateLimiter


# ── Layer protocol ─────────────────────────────────────────────────────────────
@dataclass
class BusLayer:
    """A single composable guardrail layer."""
    name: str
    check_fn: Callable[..., Any]
    enabled: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class BusResult:
    """Result from the GuardrailBus."""
    allowed: bool
    decision: str          # ALLOW | BLOCK | REDIRECT | THROTTLE
    category: str
    severity: str
    sanitized_text: str
    user_message: str
    layers_passed: list[str] = field(default_factory=list)
    layers_failed: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "decision": self.decision,
            "category": self.category,
            "severity": self.severity,
            "sanitized_text": self.sanitized_text,
            "user_message": self.user_message,
            "layers_passed": self.layers_passed,
            "layers_failed": self.layers_failed,
            "metadata": self.metadata,
        }


class GuardrailBus:
    """Unified composable guardrail pipeline.

    Usage:
        bus = GuardrailBus()
        result = bus.check_input(user_query)
        if not result.allowed:
            return result.user_message
    """

    def __init__(self) -> None:
        self._input_layers: list[BusLayer] = []
        self._output_layers: list[BusLayer] = []
        self._rate_limiter = RateLimiter(max_calls=30, window_seconds=60)
        self._setup_default_layers()

    def _setup_default_layers(self) -> None:
        """Register the default input and output guardrail layers."""
        # ── Input layers ───────────────────────────────────────────────────────
        self.add_input_layer(BusLayer(
            name="structural_check",
            check_fn=self._structural_check,
            tags=["structural", "core"],
        ))
        self.add_input_layer(BusLayer(
            name="rate_limit",
            check_fn=self._rate_limit_check,
            tags=["rate_limit", "abuse"],
        ))
        self.add_input_layer(BusLayer(
            name="attack_patterns",
            check_fn=self._attack_pattern_check,
            tags=["security", "core"],
        ))
        self.add_input_layer(BusLayer(
            name="domain_policy",
            check_fn=self._domain_policy_check,
            tags=["policy", "core"],
        ))
        self.add_input_layer(BusLayer(
            name="pii_redaction",
            check_fn=self._pii_redact,
            tags=["pii", "privacy"],
        ))

        # ── Output layers ──────────────────────────────────────────────────────
        self.add_output_layer(BusLayer(
            name="output_safety",
            check_fn=self._output_safety_check,
            tags=["security", "core"],
        ))
        self.add_output_layer(BusLayer(
            name="output_pii",
            check_fn=self._output_pii_check,
            tags=["pii", "privacy"],
        ))

    def add_input_layer(self, layer: BusLayer) -> None:
        self._input_layers.append(layer)

    def add_output_layer(self, layer: BusLayer) -> None:
        self._output_layers.append(layer)

    def enable_layer(self, name: str, enabled: bool = True) -> None:
        for layer in self._input_layers + self._output_layers:
            if layer.name == name:
                layer.enabled = enabled
                audit_event("guardrail_bus", "LAYER_TOGGLED", name, f"enabled={enabled}")

    # ── Input layer implementations ────────────────────────────────────────────
    def _structural_check(self, text: str) -> BusResult | None:
        """Delegate to the existing input guard for structural + attack checks."""
        result: GuardResult = check_query(text)
        if not result.allowed:
            return BusResult(
                allowed=False,
                decision=result.decision,
                category=result.category,
                severity=result.severity,
                sanitized_text="",
                user_message=result.user_message,
                layers_failed=["structural_check"],
                metadata={"reasons": result.reasons},
            )
        return None  # pass-through

    def _attack_pattern_check(self, text: str) -> BusResult | None:
        """Already covered by structural_check (delegates to check_query)."""
        return None  # handled above

    def _domain_policy_check(self, text: str) -> BusResult | None:
        """Already covered by structural_check (delegates to check_query)."""
        return None

    def _rate_limit_check(self, text: str) -> BusResult | None:
        if not self._rate_limiter.allow():
            return BusResult(
                allowed=False,
                decision="THROTTLE",
                category="rate_limit",
                severity="MEDIUM",
                sanitized_text="",
                user_message="Too many requests. Please wait a moment before asking again.",
                layers_failed=["rate_limit"],
                metadata={"retry_after_s": self._rate_limiter.retry_after()},
            )
        return None

    def _pii_redact(self, text: str) -> BusResult | None:
        """PII redaction — always passes but annotates the sanitized text."""
        return None  # Handled inline in check_input()

    # ── Output layer implementations ───────────────────────────────────────────
    def _output_safety_check(self, text: str, facts: Iterable | None = None) -> OutputCheck:
        return check_output(text, facts)

    def _output_pii_check(self, text: str, facts: Iterable | None = None) -> OutputCheck:
        cleaned, redactions = redact_pii(text)
        cleaned_list = [f"pii_redacted:{k}" for k in redactions] if redactions else []
        return OutputCheck(ok=True, text=cleaned, violations=[], cleaned=cleaned_list)

    # ── Public API ─────────────────────────────────────────────────────────────
    def check_input(self, text: str) -> BusResult:
        """Run all enabled input layers. Returns on first failure."""
        started = time.perf_counter()
        passed: list[str] = []

        for layer in self._input_layers:
            if not layer.enabled:
                continue
            try:
                result = layer.check_fn(text)
                if result is not None:
                    # Layer returned a failure
                    audit_event(
                        "guardrail_bus", result.decision,
                        layer.name, result.category,
                    )
                    return result
                passed.append(layer.name)
            except Exception as exc:
                audit_event("guardrail_bus", "LAYER_ERROR", layer.name, str(exc)[:200])
                # On layer error, fail safe — block the request
                return BusResult(
                    allowed=False,
                    decision="BLOCK",
                    category="guardrail_error",
                    severity="HIGH",
                    sanitized_text="",
                    user_message="Input validation failed. Please try again.",
                    layers_failed=[layer.name],
                    metadata={"error": str(exc)[:200]},
                )

        # All layers passed — apply PII redaction
        sanitized, redactions = redact_pii(text)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        audit_event("guardrail_bus", "ALLOW", "input", f"layers={passed} latency={latency_ms}ms")
        return BusResult(
            allowed=True,
            decision="ALLOW",
            category="pii_redacted" if redactions else "clean",
            severity="LOW" if redactions else "NONE",
            sanitized_text=sanitized,
            user_message="",
            layers_passed=passed,
            metadata={"redactions": redactions, "latency_ms": latency_ms},
        )

    def check_output(self, text: str, facts: Iterable | None = None) -> BusResult:
        """Run all enabled output layers."""
        started = time.perf_counter()
        current_text = text
        passed: list[str] = []

        for layer in self._output_layers:
            if not layer.enabled:
                continue
            try:
                check: OutputCheck = layer.check_fn(current_text, facts)
                if not check.ok:
                    audit_event("guardrail_bus", "OUTPUT_BLOCKED", layer.name, str(check.violations)[:200])
                    return BusResult(
                        allowed=False,
                        decision="BLOCK",
                        category="output_violation",
                        severity="HIGH",
                        sanitized_text="",
                        user_message="",
                        layers_failed=[layer.name],
                        metadata={"violations": check.violations},
                    )
                current_text = check.text
                passed.append(layer.name)
            except Exception as exc:
                audit_event("guardrail_bus", "LAYER_ERROR", layer.name, str(exc)[:200])
                # Fail safe on output check error — block the output
                return BusResult(
                    allowed=False,
                    decision="BLOCK",
                    category="guardrail_error",
                    severity="HIGH",
                    sanitized_text="",
                    user_message="",
                    layers_failed=[layer.name],
                    metadata={"error": str(exc)[:200]},
                )

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        audit_event("guardrail_bus", "OUTPUT_ALLOWED", "output", f"layers={passed}")
        return BusResult(
            allowed=True,
            decision="ALLOW",
            category="clean",
            severity="NONE",
            sanitized_text=current_text,
            user_message="",
            layers_passed=passed,
            metadata={"latency_ms": latency_ms},
        )

    def status(self) -> dict:
        """Return the current layer configuration for observability."""
        return {
            "input_layers": [
                {"name": l.name, "enabled": l.enabled, "tags": l.tags}
                for l in self._input_layers
            ],
            "output_layers": [
                {"name": l.name, "enabled": l.enabled, "tags": l.tags}
                for l in self._output_layers
            ],
        }


# ── Singleton bus instance ─────────────────────────────────────────────────────
guardrail_bus = GuardrailBus()

