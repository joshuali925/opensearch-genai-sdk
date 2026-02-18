"""Shared test fixtures for opensearch-genai-sdk tests.

Sets up a TracerProvider with InMemorySpanExporter so tests can capture
and assert on spans without a real collector.

A single TracerProvider is shared across the entire test session because
the OTEL SDK's ProxyTracer caches the real tracer on first use and does
not re-resolve when the global provider changes. By keeping one provider
alive for the whole session and clearing the exporter between tests we
avoid this caching issue entirely.
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opensearch_genai_sdk.enrichment import AggregatorSpanProcessor


# Module-level singletons, initialised once per process.
_exporter = InMemorySpanExporter()
_provider = TracerProvider(
    resource=Resource.create({"service.name": "test-service"}),
)
_provider.add_span_processor(AggregatorSpanProcessor())
_provider.add_span_processor(SimpleSpanProcessor(_exporter))

# Allow setting the global provider (only succeeds the first time the
# module is imported; that is fine because it is process-wide).
trace._TRACER_PROVIDER_SET_ONCE._done = False
trace._TRACER_PROVIDER = None
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def _clear_spans():
    """Clear the shared InMemorySpanExporter before and after every test.

    The ``autouse=True`` ensures this runs for *every* test, even those
    that don't request the ``exporter`` fixture explicitly.
    """
    _exporter.clear()
    yield
    _exporter.clear()


@pytest.fixture()
def exporter():
    """Provide the shared InMemorySpanExporter for tests that need it.

    Call ``exporter.get_finished_spans()`` after exercising the code
    under test to inspect the captured spans.
    """
    return _exporter
