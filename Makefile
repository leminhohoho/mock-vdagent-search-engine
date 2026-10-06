.PHONY: install db-up db-down migrate ingest serve test test-unit lint fmt demo docker

# Local pgvector Postgres for the database tests.
DB_CONTAINER ?= mockserp-pg
DB_PORT ?= 54329
TEST_DATABASE_URL ?= postgresql://postgres:postgres@localhost:$(DB_PORT)/postgres
# DATABASE_URL as the app sees it (.env wins over the shell).
APP_DATABASE_URL = $$(uv run python -c 'from mockserp.config import Settings; print(Settings().database_url or "")')

install:
	uv sync

db-up:
	docker run -d --rm --name $(DB_CONTAINER) -e POSTGRES_PASSWORD=postgres \
		-p $(DB_PORT):5432 pgvector/pgvector:pg17
	until docker exec $(DB_CONTAINER) pg_isready -U postgres -h 127.0.0.1 >/dev/null; do sleep 1; done

db-down:
	docker rm -f $(DB_CONTAINER)

migrate:
	supabase db push --db-url "$(APP_DATABASE_URL)"

ingest:
	uv run mockserp ingest

serve:
	uv run mockserp serve

test:
	TEST_DATABASE_URL="$(TEST_DATABASE_URL)" uv run pytest

test-unit:
	TEST_DATABASE_URL= uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

demo:
	uv run python scripts/demo.py

docker:
	docker build -t mockserp .
