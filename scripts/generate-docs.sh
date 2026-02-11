#!/usr/bin/env bash
# Generate HTML documentation from Markdown files.
# Usage: ./scripts/generate-docs.sh

set -e

cd "$(dirname "$0")/.."

echo "Generating HTML documentation from Markdown files..."
uv run python -m docs.generate_docs

echo "Done."