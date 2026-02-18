"""OTEL pipeline setup for OpenSearch AI observability.

The register() function is the single entry point for configuring
tracing. It creates a TracerProvider, sets up the exporter (with
SigV4 if needed), and auto-discovers installed instrumentor packages.

Supports both HTTP and gRPC OTLP protocols:
  - http:// or https:// → HTTP exporter
  - grpc:// → gRPC (insecure)
  - grpcs:// → gRPC (TLS)
  - Or set protocol="http" / protocol="grpc" explicitly
"""

from __future__ import annotations

import logging
import os
import sys
from importlib.metadata import entry_points
from typing import Literal, Optional
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://localhost:21890/opentelemetry/v1/traces"

# Entry point group to discover instrumentors from.
# OpenLLMetry + official OTEL instrumentors register under this group.
_INSTRUMENTOR_GROUPS = [
    "opentelemetry_instrumentor",
]


def register(
    *,
    endpoint: Optional[str] = None,
    protocol: Optional[Literal["http", "grpc"]] = None,
    project_name: Optional[str] = None,
    service_name: Optional[str] = None,
    auth: str = "auto",
    region: Optional[str] = None,
    service: str = "osis",
    batch: bool = True,
    auto_instrument: bool = True,
    exporter: Optional[SpanExporter] = None,
    set_global: bool = True,
    headers: Optional[dict] = None,
) -> TracerProvider:
    """Configure the OTEL tracing pipeline for OpenSearch.

    One-line setup that creates a TracerProvider, configures an OTLP
    exporter (with SigV4 signing for AWS endpoints), and auto-discovers
    installed instrumentor packages.

    Supports both HTTP and gRPC OTLP transport. The protocol is inferred
    from the URL scheme, or can be set explicitly:

        http:// or https:// → HTTP exporter (default)
        grpc://             → gRPC (insecure)
        grpcs://            → gRPC (TLS)

    Args:
        endpoint: OTLP endpoint URL. Defaults to OPENSEARCH_OTEL_ENDPOINT
            env var or http://localhost:21890/opentelemetry/v1/traces.
        protocol: Force "http" or "grpc". If None, inferred from URL scheme.
        project_name: Project/service name attached to all spans.
            Defaults to OPENSEARCH_PROJECT env var or "default".
        service_name: Alias for project_name.
        auth: Authentication method.
            - "auto": Detect AWS endpoints and use SigV4, plain otherwise.
            - "sigv4": Force SigV4 signing.
            - "none": No authentication.
        region: AWS region for SigV4. Auto-detected if not provided.
        service: AWS service name for SigV4 signing (default: "osis").
        batch: Use BatchSpanProcessor (True) or SimpleSpanProcessor (False).
        auto_instrument: Discover and activate installed instrumentor packages.
        exporter: Custom SpanExporter. Overrides endpoint/auth/protocol.
        set_global: Set as the global TracerProvider (default: True).
        headers: Additional headers for the exporter.

    Returns:
        The configured TracerProvider.

    Examples:
        # Self-hosted — simplest setup (HTTP)
        register()

        # AWS — SigV4 auto-detected (HTTP)
        register(endpoint="https://pipeline.us-east-1.osis.amazonaws.com/v1/traces")

        # gRPC via URL scheme
        register(endpoint="grpc://localhost:4317")

        # gRPC with TLS
        register(endpoint="grpcs://otel-collector:4317")

        # Explicit protocol override
        register(endpoint="http://localhost:4317", protocol="grpc")
    """
    endpoint = endpoint or os.environ.get("OPENSEARCH_OTEL_ENDPOINT", DEFAULT_ENDPOINT)
    name = service_name or project_name or os.environ.get("OPENSEARCH_PROJECT", "default")

    # Step 1: Create Resource (identity tag for all spans)
    resource = Resource.create({"service.name": name})

    # Step 2: Create TracerProvider
    provider = TracerProvider(resource=resource)

    # Step 3: Create Exporter
    if exporter is None:
        exporter = _create_exporter(
            endpoint=endpoint,
            protocol=protocol,
            auth=auth,
            region=region,
            service=service,
            headers=headers,
        )

    # Step 4: Add enrichment aggregator (must come before the export processor
    # so that on_end fires while the enrichment stack is still populated)
    from opensearch_genai_sdk.enrichment import AggregatorSpanProcessor

    provider.add_span_processor(AggregatorSpanProcessor())

    # Step 5: Create export processor and wire up
    if batch:
        processor = BatchSpanProcessor(exporter)
    else:
        processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Step 6: Set as global provider
    if set_global:
        trace.set_tracer_provider(provider)

    # Step 7: Auto-instrument installed libraries
    if auto_instrument:
        _auto_instrument(provider)

    logger.info(
        "OpenSearch AI tracing initialized: endpoint=%s project=%s auth=%s",
        endpoint,
        name,
        auth,
    )

    return provider


def _infer_protocol(endpoint: str, protocol: Optional[str]) -> str:
    """Determine the OTLP transport protocol from explicit setting or URL scheme."""
    if protocol:
        return protocol

    parsed = urlparse(endpoint)
    scheme = parsed.scheme.lower()

    if scheme in ("grpc", "grpcs"):
        return "grpc"

    # Default to HTTP for http://, https://, or anything else
    return "http"


def _create_exporter(
    endpoint: str,
    protocol: Optional[str],
    auth: str,
    region: Optional[str],
    service: str,
    headers: Optional[dict],
) -> SpanExporter:
    """Create the appropriate OTLP exporter based on protocol and auth."""
    resolved_protocol = _infer_protocol(endpoint, protocol)
    use_sigv4 = auth == "sigv4" or (auth == "auto" and _is_aws_endpoint(endpoint))

    if resolved_protocol == "grpc":
        return _create_grpc_exporter(endpoint, use_sigv4, region, service, headers)

    return _create_http_exporter(endpoint, use_sigv4, region, service, headers)


def _create_http_exporter(
    endpoint: str,
    use_sigv4: bool,
    region: Optional[str],
    service: str,
    headers: Optional[dict],
) -> SpanExporter:
    """Create an HTTP OTLP exporter, with optional SigV4."""
    if use_sigv4:
        from opensearch_genai_sdk.exporters import SigV4OTLPSpanExporter

        logger.info("Using SigV4 + HTTP for endpoint: %s", endpoint)
        return SigV4OTLPSpanExporter(
            endpoint=endpoint,
            service=service,
            region=region,
            headers=headers,
        )

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _create_grpc_exporter(
    endpoint: str,
    use_sigv4: bool,
    region: Optional[str],
    service: str,
    headers: Optional[dict],
) -> SpanExporter:
    """Create a gRPC OTLP exporter."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,
    )

    parsed = urlparse(endpoint)
    scheme = parsed.scheme.lower()

    # gRPC exporter takes host:port, not a full URL
    grpc_endpoint = parsed.netloc or endpoint
    insecure = scheme != "grpcs" and scheme != "https"

    if use_sigv4:
        logger.warning(
            "SigV4 + gRPC is not yet supported. Use HTTP for AWS endpoints. "
            "Falling back to plain gRPC."
        )

    logger.info("Using gRPC exporter: %s (insecure=%s)", grpc_endpoint, insecure)
    return GRPCSpanExporter(
        endpoint=grpc_endpoint,
        insecure=insecure,
        headers=headers,
    )


def _is_aws_endpoint(endpoint: str) -> bool:
    """Detect if an endpoint is an AWS-hosted service."""
    parsed = urlparse(endpoint)
    hostname = parsed.hostname or ""
    aws_patterns = [
        ".amazonaws.com",
        ".aws.dev",
        ".osis.",
        ".es.",
        ".aoss.",
    ]
    return any(pattern in hostname for pattern in aws_patterns)


def _auto_instrument(provider: TracerProvider) -> None:
    """Discover and activate installed instrumentor packages.

    Searches both the OpenTelemetry and OpenInference entry point
    groups, so instrumentors from either ecosystem are discovered.
    """
    discovered = 0
    seen_names = set()

    for group in _INSTRUMENTOR_GROUPS:
        if sys.version_info < (3, 10):
            eps = entry_points().get(group, [])
        else:
            eps = entry_points(group=group)

        for ep in eps:
            # Avoid double-instrumenting if a package registers in both groups
            if ep.name in seen_names:
                continue
            seen_names.add(ep.name)

            try:
                instrumentor_cls = ep.load()
                instrumentor = instrumentor_cls()
                instrumentor.instrument(tracer_provider=provider)
                discovered += 1
                logger.debug("Instrumented: %s (from %s)", ep.name, group)
            except Exception as exc:
                logger.debug("Skipped instrumentor %s: %s", ep.name, exc)

    if discovered == 0:
        logger.warning(
            "No instrumentor packages found. Install instrumentors to auto-trace "
            "LLM calls, e.g.: pip install opentelemetry-instrumentation-openai"
        )
    else:
        logger.info("Auto-instrumented %d libraries", discovered)
