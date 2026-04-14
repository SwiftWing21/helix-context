"""
OpenTelemetry setup for helix-context.

Makes the retrieval pipeline observable: traces per /context span, metric
histograms per tier, counters for CWoLa bucket accumulation, gauges for
chromatin + graph density. Everything degrades gracefully if the
opentelemetry packages aren't installed — helix still runs, just blind.

Usage:
    from helix_context.telemetry import setup_telemetry, meter
    app = FastAPI()
    setup_telemetry(app, service_name="helix-context")
    h = meter.create_histogram("helix_tier_contribution", unit="score")
    h.record(5.3, attributes={"tier": "pki", "shape": "project_key"})

Environment:
    HELIX_OTEL_ENABLED       - "1" to turn on, default "0"
    HELIX_OTEL_ENDPOINT      - OTLP gRPC endpoint, default "localhost:4317"
    HELIX_OTEL_INSECURE      - "1" for plain gRPC, default "1" (dev-local)
    HELIX_OTEL_SAMPLER_RATIO - trace sampler 0.0-1.0, default 1.0
    HELIX_OTEL_REDACT_QUERY  - "1" to hash query strings in spans, default "1"
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Optional

log = logging.getLogger("helix.telemetry")


# Graceful no-op stand-ins so callers can always
# `from helix_context.telemetry import tracer, meter` without a try/except.
class _NoopSpan:
    def __enter__(self): return self
    def __exit__(self, *a): return None
    def set_attribute(self, *a, **kw): pass
    def set_status(self, *a, **kw): pass
    def record_exception(self, *a, **kw): pass
    def add_event(self, *a, **kw): pass


class _NoopTracer:
    def start_as_current_span(self, *a, **kw): return _NoopSpan()
    def start_span(self, *a, **kw): return _NoopSpan()


class _NoopInstrument:
    def record(self, *a, **kw): pass
    def add(self, *a, **kw): pass
    def set(self, *a, **kw): pass


class _NoopMeter:
    def create_histogram(self, *a, **kw): return _NoopInstrument()
    def create_counter(self, *a, **kw): return _NoopInstrument()
    def create_up_down_counter(self, *a, **kw): return _NoopInstrument()
    def create_observable_gauge(self, *a, **kw): return _NoopInstrument()
    def create_gauge(self, *a, **kw): return _NoopInstrument()


tracer: Any = _NoopTracer()
meter: Any = _NoopMeter()
_initialised = False


def _redact_query(q: str) -> str:
    """Hash query text + keep first 50 chars — default privacy mode."""
    if not q:
        return ""
    digest = hashlib.sha256(q.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{q[:50]}[hash:{digest}]" if os.environ.get(
        "HELIX_OTEL_REDACT_QUERY", "1"
    ) != "0" else q


def setup_telemetry(
    app: Any = None,
    service_name: str = "helix-context",
    service_version: str = "0.4.0b",
) -> bool:
    """Initialize OTel tracer + meter providers + FastAPI auto-instrumentation.

    Returns True if telemetry was turned on, False if it was skipped (not
    enabled, or the opentelemetry packages are missing). Safe to call
    multiple times — idempotent.
    """
    global tracer, meter, _initialised
    if _initialised:
        return True
    if os.environ.get("HELIX_OTEL_ENABLED", "0") != "1":
        log.info("OTel disabled (set HELIX_OTEL_ENABLED=1 to turn on)")
        return False
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import (
            TraceIdRatioBased, ALWAYS_ON, ParentBased,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
    except ImportError:
        log.warning(
            "OTel packages not installed — "
            "`pip install opentelemetry-distro opentelemetry-exporter-otlp "
            "opentelemetry-instrumentation-fastapi`"
        )
        return False

    endpoint = os.environ.get("HELIX_OTEL_ENDPOINT", "localhost:4317")
    insecure = os.environ.get("HELIX_OTEL_INSECURE", "1") == "1"
    try:
        ratio = float(os.environ.get("HELIX_OTEL_SAMPLER_RATIO", "1.0"))
    except ValueError:
        ratio = 1.0

    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.host": os.environ.get("COMPUTERNAME", "unknown"),
    })

    sampler = ParentBased(ALWAYS_ON if ratio >= 1.0 else TraceIdRatioBased(ratio))
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
        )
    )
    trace.set_tracer_provider(tracer_provider)
    tracer = trace.get_tracer(service_name, service_version)

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint, insecure=insecure),
        export_interval_millis=15_000,
    )
    meter_provider = MeterProvider(
        resource=resource, metric_readers=[metric_reader],
    )
    metrics.set_meter_provider(meter_provider)
    meter = metrics.get_meter(service_name, service_version)

    # Auto-instrument FastAPI if an app was provided. Wraps every route
    # in a span; free latency + status metric per endpoint.
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor().instrument_app(app)
        except ImportError:
            log.warning("opentelemetry-instrumentation-fastapi missing — "
                        "FastAPI routes will not be auto-traced")

    _initialised = True
    # Promoted to WARNING so the confirmation is visible even when the root
    # logger is at the default WARNING level (uvicorn's --log-level only
    # affects uvicorn's own loggers; helix.* loggers are not auto-promoted).
    # Without this, operators can't confirm OTel is actually on.
    log.warning("OTel telemetry ON, endpoint=%s insecure=%s sampler=%.2f",
                endpoint, insecure, ratio)
    return True


def redact_query(q: str) -> str:
    """Public redaction helper for code paths that want to stamp a
    privacy-safe query attribute on a span."""
    return _redact_query(q)


# ── Lazy instrument getters ──────────────────────────────────────────
# Modules import these once. They resolve to no-op instruments when
# telemetry is off, real instruments when on. Cached so meter calls
# happen at most once per process.

_instruments: dict = {}


def tier_contribution_histogram():
    if "tier_contribution" not in _instruments:
        _instruments["tier_contribution"] = meter.create_histogram(
            "helix_tier_contribution",
            unit="score",
            description="Per-tier bonus magnitude contributed to gene_scores",
        )
    return _instruments["tier_contribution"]


def context_latency_histogram():
    if "context_latency" not in _instruments:
        _instruments["context_latency"] = meter.create_histogram(
            "helix_context_latency_seconds",
            unit="s",
            description="End-to-end /context build time",
        )
    return _instruments["context_latency"]


def cwola_bucket_counter():
    if "cwola_bucket" not in _instruments:
        _instruments["cwola_bucket"] = meter.create_counter(
            "helix_cwola_bucket_total",
            description="CWoLa log rows by bucket (A/B/pending)",
        )
    return _instruments["cwola_bucket"]


def cwola_f_gap_gauge():
    if "cwola_f_gap" not in _instruments:
        _instruments["cwola_f_gap"] = meter.create_gauge(
            "helix_cwola_f_gap_sq",
            description="(f_A - f_B)^2 — CWoLa bucket divergence (0.16 promotes PLR)",
        )
    return _instruments["cwola_f_gap"]


def harmonic_edges_counter():
    if "harmonic_edges" not in _instruments:
        _instruments["harmonic_edges"] = meter.create_gauge(
            "helix_harmonic_edges_total",
            description="Count of harmonic_links edges by provenance source",
        )
    return _instruments["harmonic_edges"]


def chromatin_state_counter():
    if "chromatin_state" not in _instruments:
        _instruments["chromatin_state"] = meter.create_gauge(
            "helix_chromatin_state_total",
            description="Gene count by chromatin state (OPEN/EUCHROMATIN/HETEROCHROMATIN)",
        )
    return _instruments["chromatin_state"]


def genome_size_gauge():
    if "genome_size" not in _instruments:
        _instruments["genome_size"] = meter.create_gauge(
            "helix_genome_size_bytes",
            unit="By",
            description="Genome total char count — raw vs compressed",
        )
    return _instruments["genome_size"]


def tier_fired_counter():
    if "tier_fired" not in _instruments:
        _instruments["tier_fired"] = meter.create_counter(
            "helix_tier_fired_total",
            description="Retrieval tier activation events, labelled by tier",
        )
    return _instruments["tier_fired"]


def hub_concentration_gauge():
    if "hub_concentration" not in _instruments:
        _instruments["hub_concentration"] = meter.create_gauge(
            "helix_hub_concentration_ratio",
            description="harmonic_links inbound-degree top-1% mean / overall mean. "
                        "Watch for condensation transition (preferential-attachment "
                        "graphs collapse flow into hubs as N grows). Healthy ≲ ~10x; "
                        "rising trend = hub monopolization, retrieval flowing through "
                        "fewer paths than the edge count suggests.",
        )
    return _instruments["hub_concentration"]


def hub_inbound_degree_gauge():
    if "hub_inbound_degree" not in _instruments:
        _instruments["hub_inbound_degree"] = meter.create_gauge(
            "helix_hub_inbound_degree",
            description="harmonic_links inbound-degree summary statistics, labelled by stat "
                        "(max / p99 / p95 / p50 / mean). Backfill cap is 500; values "
                        "approaching that consistently mean the cap is the binding constraint.",
        )
    return _instruments["hub_inbound_degree"]


def emit_gauges_snapshot(genome) -> None:
    """Poll-driven gauges for chromatin + harmonic-edges + genome size.

    Prometheus scrapes via the collector every 15s; we refresh these
    absolute-value metrics on each /stats call (cheap DB queries) so
    the dashboard gauges track live state instead of event stream.
    No-op when OTel is off — the noop instruments just drop the calls.
    """
    try:
        cur = genome.read_conn.cursor()
        # Chromatin state distribution.
        chrom = cur.execute(
            "SELECT chromatin, COUNT(*) FROM genes GROUP BY chromatin"
        ).fetchall()
        chrom_gauge = chromatin_state_counter()
        for state, n in chrom:
            label = {0: "open", 1: "euchromatin", 2: "heterochromatin"}.get(
                int(state) if state is not None else 0, "unknown",
            )
            chrom_gauge.set(int(n), {"state": label})

        # Harmonic-edges by provenance source.
        edges = cur.execute(
            "SELECT source, COUNT(*) FROM harmonic_links GROUP BY source"
        ).fetchall()
        edges_gauge = harmonic_edges_counter()
        for source, n in edges:
            edges_gauge.set(int(n), {"source": source or "unknown"})

        # Genome total-chars (raw vs compressed) — genome.stats() owns this
        # view; hand-roll here so we don't circular-import stats().
        row = cur.execute(
            "SELECT "
            "SUM(LENGTH(content)) AS raw, "
            "SUM(LENGTH(complement)) AS compressed "
            "FROM genes WHERE chromatin=0"
        ).fetchone()
        size_gauge = genome_size_gauge()
        if row and row[0]:
            size_gauge.set(int(row[0]), {"kind": "raw"})
        if row and row[1]:
            size_gauge.set(int(row[1]), {"kind": "compressed"})

        # Hub-concentration / inbound-degree summary. Preferential-attachment
        # graphs have no classical percolation threshold but condense flow into
        # hubs as N grows; the right order parameter for that pathology is the
        # ratio of top-1% inbound degree to mean inbound degree, not the
        # giant-component size. Backfill caps inbound at 500 (see
        # scripts/backfill_seeded_edges.py); persistent values near the cap are
        # the cap acting as the binding constraint, not organic structure.
        in_degrees = [
            int(n) for (_, n) in cur.execute(
                "SELECT gene_id_b, COUNT(*) FROM harmonic_links GROUP BY gene_id_b"
            ).fetchall()
        ]
        if in_degrees:
            in_degrees.sort()
            n = len(in_degrees)
            mean_deg = sum(in_degrees) / n
            top_1pct_count = max(1, n // 100)
            top_1pct_mean = sum(in_degrees[-top_1pct_count:]) / top_1pct_count
            ratio = top_1pct_mean / mean_deg if mean_deg > 0 else 0.0

            hub_concentration_gauge().set(float(ratio))
            deg_gauge = hub_inbound_degree_gauge()
            deg_gauge.set(float(in_degrees[-1]),               {"stat": "max"})
            deg_gauge.set(float(in_degrees[int(n * 0.99) - 1]), {"stat": "p99"})
            deg_gauge.set(float(in_degrees[int(n * 0.95) - 1]), {"stat": "p95"})
            deg_gauge.set(float(in_degrees[n // 2]),            {"stat": "p50"})
            deg_gauge.set(float(mean_deg),                     {"stat": "mean"})
    except Exception:
        # Promoted from debug to warning: silent debug-level was hiding a
        # real failure (chromatin gauge would emit, harmonic/hub/genome_size
        # would silently disappear). If you see this in normal operation,
        # the SQL inside this function raised — likely a stale read_conn
        # schema cache or a replica-vs-master path mismatch.
        log.warning("emit_gauges_snapshot failed", exc_info=True)
