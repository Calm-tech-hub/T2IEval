default:
    @just --list

setup:
    uv sync
    uv run pre-commit install

lint:
    uv run ruff check . --fix
    uv run ruff format .
    uv run pyright src

test:
    uv run pytest

run *args:
    uv run t2i-eval {{args}}

build:
    uv build

publish:
    uv build
    uv publish

clean:
    rm -rf .pytest_cache .ruff_cache .venv dist