#!/bin/bash
set -e

uv run ruff format
uv run mypy .
uv run ruff check --fix
uv run pytest
