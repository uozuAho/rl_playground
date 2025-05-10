#!/bin/bash
set -e

uv run mypy .
uv run ruff check --fix
uv run ruff format
