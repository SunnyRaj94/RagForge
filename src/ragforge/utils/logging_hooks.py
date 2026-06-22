import time
import functools
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import mlflow

# Configure OpenTelemetry to export to Arize Phoenix
from ragforge.config import PHOENIX_COLLECTOR_URL

phoenix_url = PHOENIX_COLLECTOR_URL

try:
    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=phoenix_url))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
except Exception as e:
    # If already initialized or fails, fallback to current tracer provider
    pass

tracer = trace.get_tracer("ragforge-pipeline")


def observe_stage(stage_name: str):
    """
    A decorator that logs metrics to MLflow and exports tracer spans to Arize Phoenix.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start OpenTelemetry Span
            with tracer.start_as_current_span(stage_name) as span:
                start_time = time.perf_counter()

                # Tag inputs in OpenTelemetry attributes
                span.set_attribute("input.arguments", str(args)[:1000])
                if "model_id" in kwargs:
                    span.set_attribute("embedding.model", kwargs["model_id"])
                    try:
                        mlflow.log_param("embedding_model", kwargs["model_id"])
                    except Exception:
                        pass

                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("status", "SUCCESS")
                    return result
                except Exception as e:
                    span.set_attribute("status", "FAILED")
                    span.record_exception(e)
                    try:
                        mlflow.log_param(f"{stage_name}_error", str(e)[:250])
                    except Exception:
                        pass
                    raise e
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    # Log metric to MLflow (if active run exists)
                    try:
                        if mlflow.active_run():
                            mlflow.log_metric(f"{stage_name}_latency_ms", duration_ms)
                    except Exception:
                        pass

        return wrapper

    return decorator
