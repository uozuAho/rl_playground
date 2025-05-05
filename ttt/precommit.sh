#!/bin/bash
set -e

mypy
uv run ruff check
uv run pytest
