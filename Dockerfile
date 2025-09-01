ARG BASE_IMAGE=ghcr.io/astral-sh/uv:python3.13-trixie-slim

FROM $BASE_IMAGE AS build

WORKDIR /app

# Leverage caching for dependency resolution
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy application source
COPY api ./api
COPY web ./web
COPY *.py ./

FROM $BASE_IMAGE
WORKDIR /app
COPY knowledge_base ./knowledge_base

# Copy project files and virtualenv from build stage
COPY --from=build /app /app

# Set environment variables for runtime
ARG PORT
ENV PORT=${PORT:-8080}
ENV HOST=0.0.0.0

# Expose port
EXPOSE $PORT

# Default command; pass env via docker run --env-file or env vars in deployment
CMD uv run uvicorn api.main:socket_app --host 0.0.0.0 --port $PORT


