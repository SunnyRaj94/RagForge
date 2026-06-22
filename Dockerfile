FROM python:3.12-slim-bookworm AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project configuration files
COPY pyproject.toml uv.lock ./

# Sync dependencies (without installing the project itself yet)
RUN uv sync --frozen --no-install-project

# Copy source code and install project
COPY src/ ./src/
COPY servers/ ./servers/
RUN uv sync --frozen

FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies (curl for health check, easyocr/Tesseract requirements if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv containing dependencies and the source package
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/servers /app/servers

# Set path and pythonpath so dependencies are visible
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "src.ragforge.api:app", "--host", "0.0.0.0", "--port", "8000"]
