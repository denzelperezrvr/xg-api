FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY models/ models/
RUN uv sync --frozen --no-dev

EXPOSE 8080
CMD ["sh", "-c", "uvicorn xg.api:app --host 0.0.0.0 --port ${PORT:-8080}"]