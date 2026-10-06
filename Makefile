.PHONY: install ingest serve test lint fmt demo

install:
	uv sync

ingest:
	uv run mockserp ingest

serve:
	uv run mockserp serve

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

demo:
	uv run python scripts/demo.py
