#!/bin/bash
set -e

uv run mypy .
uv run ruff check
uv run ruff format
