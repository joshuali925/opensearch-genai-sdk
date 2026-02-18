"""Span-ID-based enrichment for lifting child gen_ai attributes to parent spans.

When auto-instrumentors (e.g. opentelemetry-instrumentation-openai) create child
spans with gen_ai.* attributes, those attributes only live on the child span.
This module provides an AggregatorSpanProcessor that propagates gen_ai.*
attributes from any child span to its live parent span — regardless of whether
the parent is a decorator span, an ASGI span, or anything else.

How it works:

1. on_start — store every live Span object by span_id
2. on_end   — read the ending span's gen_ai.* attributes, look up the
              live parent span, and set the accumulated values on it
3. When the parent itself ends, it already carries the aggregated
   gen_ai.* attributes and they propagate further up the tree

Aggregation strategies:
  - gen_ai.usage.input_tokens / output_tokens  → SUM
  - gen_ai.request.model / response.model      → last-write-wins
  - gen_ai.provider.name                       → last-write-wins
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

# Attributes that are summed across child spans
_SUM_ATTRIBUTES = frozenset(
    {
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    }
)

# Attributes where last-write-wins
_LWW_ATTRIBUTES = frozenset(
    {
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.provider.name",
    }
)


def _get_attribute(span: trace.Span, key: str) -> object | None:
    """Read a single attribute from a live SDK Span."""
    sdk_attrs = getattr(span, "_attributes", None)
    if sdk_attrs is not None:
        return sdk_attrs.get(key)
    return None


@dataclass
class _SpanAccumulator:
    """Thread-safe accumulator for a single parent span's enrichment.

    Tracks accumulated sums and LWW values from children.  Also remembers
    which LWW attributes were already present on the parent *before* the
    first child contributed (user-set attributes) so they are not
    overwritten.
    """

    _sums: dict[str, int] = field(default_factory=dict)
    _lww: dict[str, str] = field(default_factory=dict)
    _user_set_lww: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_and_apply(
        self, child: ReadableSpan, parent: trace.Span
    ) -> None:
        """Atomically accumulate child attrs and set them on the parent."""
        child_attrs = child.attributes
        if not child_attrs:
            return

        with self._lock:
            changed = False

            for key in _SUM_ATTRIBUTES:
                val = child_attrs.get(key)
                if val is not None:
                    self._sums[key] = self._sums.get(key, 0) + int(val)
                    changed = True

            for key in _LWW_ATTRIBUTES:
                val = child_attrs.get(key)
                if val is not None:
                    # First time seeing this key — check if the parent
                    # already had it set (user-set), and protect it.
                    if key not in self._lww and key not in self._user_set_lww:
                        if _get_attribute(parent, key) is not None:
                            self._user_set_lww.add(key)
                            continue
                    if key in self._user_set_lww:
                        continue
                    self._lww[key] = str(val)
                    changed = True

            if changed:
                for key, val in self._sums.items():
                    parent.set_attribute(key, val)
                for key, val in self._lww.items():
                    parent.set_attribute(key, val)


class AggregatorSpanProcessor(SpanProcessor):
    """SpanProcessor that propagates gen_ai.* attributes from children to parents.

    Tracks every live span via on_start / on_end.  When a child span ends,
    its gen_ai.* attributes are accumulated into the parent's accumulator
    and immediately applied to the live parent Span.  This works for *any*
    parent span — decorator spans, ASGI server spans, HTTPX client spans, etc.

    Must be added to the TracerProvider *before* the export processor so that
    on_end fires while parent spans are still alive.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # span_id → live (mutable) Span object
        self._live_spans: dict[int, trace.Span] = {}
        # span_id → accumulator collecting children's gen_ai.* contributions
        self._accumulators: dict[int, _SpanAccumulator] = {}

    def on_start(
        self,
        span: Span,
        parent_context: Optional[trace.Context] = None,
    ) -> None:
        span_id = span.context.span_id
        with self._lock:
            self._live_spans[span_id] = span
            self._accumulators[span_id] = _SpanAccumulator()

    def on_end(self, span: ReadableSpan) -> None:
        span_id = span.context.span_id

        # Clean up this span's bookkeeping
        with self._lock:
            self._live_spans.pop(span_id, None)
            self._accumulators.pop(span_id, None)

        # Propagate gen_ai.* attributes to the live parent span
        parent_ctx = span.parent
        if parent_ctx is None or parent_ctx.is_remote:
            return

        parent_id = parent_ctx.span_id
        with self._lock:
            parent_span = self._live_spans.get(parent_id)
            parent_acc = self._accumulators.get(parent_id)

        if parent_span is not None and parent_acc is not None:
            parent_acc.record_and_apply(span, parent_span)

    def shutdown(self) -> None:
        with self._lock:
            self._live_spans.clear()
            self._accumulators.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
