#!/bin/bash
mypy .
uv run ruff check
