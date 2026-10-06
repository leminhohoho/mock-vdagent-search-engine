# Stateless mockserp server: configure with environment variables only
# (DATABASE_URL, EMBEDDING_API_KEY, MOCK_API_KEY, ...). PORT defaults to 8000.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN useradd --system --uid 10001 --no-create-home mockserp
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH HOST=0.0.0.0 PORT=8000 PYTHONUNBUFFERED=1
# Pin "now" so time_range filters give the same results on any day; override with -e MOCK_NOW=...
ENV MOCK_NOW=2026-10-06T00:00:00Z
WORKDIR /app
USER mockserp
EXPOSE 8000
CMD ["mockserp", "serve"]
