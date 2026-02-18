"""Tests for opensearch_genai_sdk.enrichment.

Covers the span-ID-based enrichment mechanism that lifts child
gen_ai.* attributes up to parent spans — including non-decorator
parents like ASGI server spans.
"""

import asyncio

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan

from opensearch_genai_sdk.decorators import agent, task, tool, workflow


# ---------------------------------------------------------------------------
# Helpers — simulate an LLM child span with gen_ai.* attributes
# ---------------------------------------------------------------------------

_tracer = trace.get_tracer("test-enrichment")


def _make_llm_span(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "gpt-4",
    response_model: str | None = None,
    provider: str = "openai",
) -> None:
    """Create and immediately end a child span with gen_ai attributes,
    simulating what an auto-instrumentor would do."""
    with _tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.provider.name", provider)
        if response_model is not None:
            span.set_attribute("gen_ai.response.model", response_model)


async def _make_llm_span_async(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "gpt-4",
    response_model: str | None = None,
    provider: str = "openai",
) -> None:
    """Async variant of _make_llm_span."""
    with _tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.provider.name", provider)
        if response_model is not None:
            span.set_attribute("gen_ai.response.model", response_model)


def _find_span(spans: list[ReadableSpan], name: str) -> ReadableSpan:
    """Find a span by name in a list of finished spans."""
    for s in spans:
        if s.name == name:
            return s
    raise ValueError(f"No span named {name!r} in {[s.name for s in spans]}")


# ---------------------------------------------------------------------------
# Single LLM child → parent gets tokens + model + provider
# ---------------------------------------------------------------------------


class TestSingleChildEnrichment:

    def test_single_llm_child_enriches_parent(self, exporter):
        @agent(name="single_child_agent")
        def my_agent():
            _make_llm_span(input_tokens=100, output_tokens=50, model="gpt-4", provider="openai")
            return "done"

        my_agent()
        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "single_child_agent")

        assert parent.attributes["gen_ai.usage.input_tokens"] == 100
        assert parent.attributes["gen_ai.usage.output_tokens"] == 50
        assert parent.attributes["gen_ai.request.model"] == "gpt-4"
        assert parent.attributes["gen_ai.provider.name"] == "openai"

    def test_response_model_propagated(self, exporter):
        @tool(name="resp_model_tool")
        def my_tool():
            _make_llm_span(
                input_tokens=10,
                output_tokens=5,
                model="gpt-4",
                response_model="gpt-4-0613",
                provider="openai",
            )
            return "done"

        my_tool()
        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "resp_model_tool")

        assert parent.attributes["gen_ai.response.model"] == "gpt-4-0613"


# ---------------------------------------------------------------------------
# Multiple LLM children → tokens summed, model last-write-wins
# ---------------------------------------------------------------------------


class TestMultipleChildrenEnrichment:

    def test_tokens_summed_across_children(self, exporter):
        @agent(name="multi_child_agent")
        def my_agent():
            _make_llm_span(input_tokens=100, output_tokens=50, model="gpt-4")
            _make_llm_span(input_tokens=200, output_tokens=75, model="gpt-4")
            return "done"

        my_agent()
        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "multi_child_agent")

        assert parent.attributes["gen_ai.usage.input_tokens"] == 300
        assert parent.attributes["gen_ai.usage.output_tokens"] == 125

    def test_model_last_write_wins(self, exporter):
        @agent(name="lww_agent")
        def my_agent():
            _make_llm_span(input_tokens=100, output_tokens=50, model="gpt-3.5-turbo")
            _make_llm_span(input_tokens=200, output_tokens=75, model="gpt-4")
            return "done"

        my_agent()
        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "lww_agent")

        assert parent.attributes["gen_ai.request.model"] == "gpt-4"


# ---------------------------------------------------------------------------
# No LLM children → no enrichment attributes added
# ---------------------------------------------------------------------------


class TestNoChildrenEnrichment:

    def test_no_enrichment_when_no_llm_children(self, exporter):
        @agent(name="no_child_agent")
        def my_agent():
            return "done"

        my_agent()
        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "no_child_agent")

        assert "gen_ai.usage.input_tokens" not in parent.attributes
        assert "gen_ai.usage.output_tokens" not in parent.attributes
        assert "gen_ai.request.model" not in parent.attributes
        assert "gen_ai.provider.name" not in parent.attributes


# ---------------------------------------------------------------------------
# Nested decorators: agent → tool → LLM — both agent and tool enriched
# ---------------------------------------------------------------------------


class TestNestedDecorators:

    def test_agent_tool_llm_nesting(self, exporter):
        @tool(name="inner_tool")
        def my_tool():
            _make_llm_span(input_tokens=100, output_tokens=50, model="gpt-4")
            return "tool result"

        @agent(name="outer_agent")
        def my_agent():
            return my_tool()

        my_agent()
        spans = exporter.get_finished_spans()

        tool_span = _find_span(spans, "inner_tool")
        agent_span = _find_span(spans, "outer_agent")

        # Tool span gets direct enrichment from LLM child
        assert tool_span.attributes["gen_ai.usage.input_tokens"] == 100
        assert tool_span.attributes["gen_ai.usage.output_tokens"] == 50

        # Agent span gets enrichment from the tool span (which was enriched by LLM)
        assert agent_span.attributes["gen_ai.usage.input_tokens"] == 100
        assert agent_span.attributes["gen_ai.usage.output_tokens"] == 50

    def test_multiple_tools_under_one_agent(self, exporter):
        @tool(name="tool_a")
        def tool_a():
            _make_llm_span(input_tokens=100, output_tokens=50, model="gpt-4")
            return "a"

        @tool(name="tool_b")
        def tool_b():
            _make_llm_span(input_tokens=200, output_tokens=75, model="gpt-4o")
            return "b"

        @agent(name="multi_tool_agent")
        def my_agent():
            tool_a()
            tool_b()
            return "done"

        my_agent()
        spans = exporter.get_finished_spans()

        tool_a_span = _find_span(spans, "tool_a")
        tool_b_span = _find_span(spans, "tool_b")
        agent_span = _find_span(spans, "multi_tool_agent")

        # Each tool gets its own LLM's tokens
        assert tool_a_span.attributes["gen_ai.usage.input_tokens"] == 100
        assert tool_b_span.attributes["gen_ai.usage.input_tokens"] == 200

        # Agent gets sum from both tool spans (each enriched with their LLM tokens)
        assert agent_span.attributes["gen_ai.usage.input_tokens"] == 300
        assert agent_span.attributes["gen_ai.usage.output_tokens"] == 125

        # Last-write-wins for model on agent span (tool_b ended last)
        assert agent_span.attributes["gen_ai.request.model"] == "gpt-4o"

    def test_three_level_nesting_workflow_agent_tool_llm(self, exporter):
        @tool(name="deep_tool")
        def my_tool():
            _make_llm_span(input_tokens=50, output_tokens=25, model="claude-3")
            return "tool done"

        @agent(name="mid_agent")
        def my_agent():
            return my_tool()

        @workflow(name="top_workflow")
        def my_workflow():
            return my_agent()

        my_workflow()
        spans = exporter.get_finished_spans()

        tool_span = _find_span(spans, "deep_tool")
        agent_span = _find_span(spans, "mid_agent")
        wf_span = _find_span(spans, "top_workflow")

        # All three levels should have the enrichment
        for span in (tool_span, agent_span, wf_span):
            assert span.attributes["gen_ai.usage.input_tokens"] == 50
            assert span.attributes["gen_ai.usage.output_tokens"] == 25
            assert span.attributes["gen_ai.request.model"] == "claude-3"


# ---------------------------------------------------------------------------
# Error path — enrichment still applied
# ---------------------------------------------------------------------------


class TestErrorPathEnrichment:

    def test_enrichment_applied_on_error(self, exporter):
        @agent(name="error_agent")
        def my_agent():
            _make_llm_span(input_tokens=100, output_tokens=50, model="gpt-4")
            raise ValueError("something broke")

        with pytest.raises(ValueError, match="something broke"):
            my_agent()

        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "error_agent")

        # Enrichment should still be applied even though the function errored
        assert parent.attributes["gen_ai.usage.input_tokens"] == 100
        assert parent.attributes["gen_ai.usage.output_tokens"] == 50
        assert parent.attributes["gen_ai.request.model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Manual attribute not overwritten
# ---------------------------------------------------------------------------


class TestManualAttributePreservation:

    def test_manual_attribute_not_overwritten(self, exporter):
        @agent(name="manual_attr_agent")
        def my_agent():
            # Manually set model before LLM child creates its span
            span = trace.get_current_span()
            span.set_attribute("gen_ai.request.model", "my-custom-model")
            _make_llm_span(input_tokens=100, output_tokens=50, model="gpt-4")
            return "done"

        my_agent()
        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "manual_attr_agent")

        # Manual attribute should be preserved (not overwritten by enrichment)
        assert parent.attributes["gen_ai.request.model"] == "my-custom-model"
        # But summed attributes should still be set (they weren't manually set)
        assert parent.attributes["gen_ai.usage.input_tokens"] == 100


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


class TestAsyncEnrichment:

    @pytest.mark.asyncio
    async def test_async_agent_enrichment(self, exporter):
        @agent(name="async_enriched_agent")
        async def my_agent():
            await _make_llm_span_async(input_tokens=150, output_tokens=60, model="gpt-4")
            return "async done"

        await my_agent()
        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "async_enriched_agent")

        assert parent.attributes["gen_ai.usage.input_tokens"] == 150
        assert parent.attributes["gen_ai.usage.output_tokens"] == 60
        assert parent.attributes["gen_ai.request.model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_async_nested_enrichment(self, exporter):
        @tool(name="async_inner_tool")
        async def my_tool():
            await _make_llm_span_async(input_tokens=80, output_tokens=40, model="gpt-4o")
            return "tool done"

        @agent(name="async_outer_agent")
        async def my_agent():
            return await my_tool()

        await my_agent()
        spans = exporter.get_finished_spans()

        tool_span = _find_span(spans, "async_inner_tool")
        agent_span = _find_span(spans, "async_outer_agent")

        assert tool_span.attributes["gen_ai.usage.input_tokens"] == 80
        assert agent_span.attributes["gen_ai.usage.input_tokens"] == 80

    @pytest.mark.asyncio
    async def test_async_error_enrichment(self, exporter):
        @agent(name="async_error_agent")
        async def my_agent():
            await _make_llm_span_async(input_tokens=100, output_tokens=50, model="gpt-4")
            raise RuntimeError("async error")

        with pytest.raises(RuntimeError, match="async error"):
            await my_agent()

        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "async_error_agent")

        assert parent.attributes["gen_ai.usage.input_tokens"] == 100


# ---------------------------------------------------------------------------
# Generator variants
# ---------------------------------------------------------------------------


class TestGeneratorEnrichment:

    def test_sync_generator_enrichment(self, exporter):
        @tool(name="gen_enriched_tool")
        def my_tool():
            _make_llm_span(input_tokens=100, output_tokens=50, model="gpt-4")
            yield "a"
            yield "b"

        items = list(my_tool())
        assert items == ["a", "b"]

        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "gen_enriched_tool")

        assert parent.attributes["gen_ai.usage.input_tokens"] == 100
        assert parent.attributes["gen_ai.usage.output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_async_generator_enrichment(self, exporter):
        @tool(name="async_gen_enriched_tool")
        async def my_tool():
            await _make_llm_span_async(input_tokens=120, output_tokens=60, model="gpt-4")
            yield "x"
            yield "y"

        items = []
        async for item in my_tool():
            items.append(item)
        assert items == ["x", "y"]

        spans = exporter.get_finished_spans()
        parent = _find_span(spans, "async_gen_enriched_tool")

        assert parent.attributes["gen_ai.usage.input_tokens"] == 120
        assert parent.attributes["gen_ai.usage.output_tokens"] == 60


# ---------------------------------------------------------------------------
# Non-decorator parent spans (e.g. ASGI server spans)
# ---------------------------------------------------------------------------


class TestNonDecoratorParentEnrichment:
    """Verify that enrichment propagates to ANY parent span, not just
    decorator spans.  This simulates the ASGI / HTTPX case where an
    auto-instrumentor creates a parent span and the SDK decorator
    creates a child."""

    def test_plain_parent_gets_enrichment_from_decorator_child(self, exporter):
        """Simulates: ASGI span → @agent span → LLM span.
        The plain 'server_span' (non-decorator) should get gen_ai attrs."""
        with _tracer.start_as_current_span("server_span"):
            @agent(name="decorated_agent")
            def my_agent():
                _make_llm_span(input_tokens=100, output_tokens=50, model="gpt-4")
                return "done"

            my_agent()

        spans = exporter.get_finished_spans()
        server = _find_span(spans, "server_span")
        agent_span = _find_span(spans, "decorated_agent")

        # Decorator span gets enrichment from LLM child
        assert agent_span.attributes["gen_ai.usage.input_tokens"] == 100

        # Non-decorator parent also gets enrichment
        assert server.attributes["gen_ai.usage.input_tokens"] == 100
        assert server.attributes["gen_ai.usage.output_tokens"] == 50
        assert server.attributes["gen_ai.request.model"] == "gpt-4"
        assert server.attributes["gen_ai.provider.name"] == "openai"

    def test_plain_parent_sums_from_multiple_decorator_children(self, exporter):
        """ASGI span → @task A (100 tokens) + @task B (200 tokens).
        The ASGI span should have 300 total tokens."""
        with _tracer.start_as_current_span("server_span"):
            @task(name="task_a")
            def task_a():
                _make_llm_span(input_tokens=100, output_tokens=40, model="gpt-4")
                return "a"

            @task(name="task_b")
            def task_b():
                _make_llm_span(input_tokens=200, output_tokens=80, model="gpt-4o")
                return "b"

            task_a()
            task_b()

        spans = exporter.get_finished_spans()
        server = _find_span(spans, "server_span")

        assert server.attributes["gen_ai.usage.input_tokens"] == 300
        assert server.attributes["gen_ai.usage.output_tokens"] == 120
        assert server.attributes["gen_ai.request.model"] == "gpt-4o"  # LWW

    @pytest.mark.asyncio
    async def test_async_plain_parent_gets_enrichment(self, exporter):
        """Async variant: non-decorator parent with async decorator child."""
        with _tracer.start_as_current_span("async_server_span"):
            @agent(name="async_decorated_agent")
            async def my_agent():
                await _make_llm_span_async(input_tokens=150, output_tokens=60, model="claude-3")
                return "done"

            await my_agent()

        spans = exporter.get_finished_spans()
        server = _find_span(spans, "async_server_span")

        assert server.attributes["gen_ai.usage.input_tokens"] == 150
        assert server.attributes["gen_ai.usage.output_tokens"] == 60
        assert server.attributes["gen_ai.request.model"] == "claude-3"

    def test_four_level_nesting_plain_parent_workflow_agent_tool_llm(self, exporter):
        """Plain parent → @workflow → @agent → @tool → LLM.
        All four levels should have enrichment."""
        with _tracer.start_as_current_span("root_server"):
            @tool(name="deep_tool_4l")
            def my_tool():
                _make_llm_span(input_tokens=50, output_tokens=25, model="gpt-4")
                return "done"

            @agent(name="mid_agent_4l")
            def my_agent():
                return my_tool()

            @workflow(name="top_workflow_4l")
            def my_workflow():
                return my_agent()

            my_workflow()

        spans = exporter.get_finished_spans()
        root = _find_span(spans, "root_server")
        wf = _find_span(spans, "top_workflow_4l")
        ag = _find_span(spans, "mid_agent_4l")
        tl = _find_span(spans, "deep_tool_4l")

        for span in (tl, ag, wf, root):
            assert span.attributes["gen_ai.usage.input_tokens"] == 50
            assert span.attributes["gen_ai.usage.output_tokens"] == 25
            assert span.attributes["gen_ai.request.model"] == "gpt-4"
