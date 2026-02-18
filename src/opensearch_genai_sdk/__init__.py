"""OpenSearch AI Observability SDK.

OTEL-native tracing, scoring, and evaluation for LLM applications.
"""

from opensearch_genai_sdk.register import register
from opensearch_genai_sdk.decorators import workflow, task, agent, tool
from opensearch_genai_sdk.enrichment import AggregatorSpanProcessor
from opensearch_genai_sdk.score import score
from opensearch_genai_sdk.evals import evaluate, Scorer, Score

__all__ = [
    # Setup
    "register",
    # Decorators
    "workflow",
    "task",
    "agent",
    "tool",
    # Enrichment
    "AggregatorSpanProcessor",
    # Scoring
    "score",
    # Evals
    "evaluate",
    "Scorer",
    "Score",
]
