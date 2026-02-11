#!/usr/bin/env python3
"""
Generate HTML documentation from Markdown files.

This script converts all Markdown files in the docs/ directory to HTML
using the existing convert_to_html.py script. It only regenerates files
when the Markdown source is newer than the HTML output or when HTML
is missing.
"""

import sys
import os
from pathlib import Path
import subprocess
from datetime import datetime

def should_regenerate(md_path: Path, html_path: Path) -> bool:
    """Return True if HTML needs to be regenerated."""
    if not html_path.exists():
        return True
    md_mtime = md_path.stat().st_mtime
    html_mtime = html_path.stat().st_mtime
    return md_mtime > html_mtime

def convert_markdown_to_html(md_path: Path, html_path: Path = None):
    """Convert a single Markdown file to HTML using convert_to_html.py."""
    # Use the existing conversion script
    cmd = ["uv", "run", "python", "docs/convert_to_html.py", str(md_path)]
    if html_path:
        cmd.append(str(html_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        if result.stderr:
            print(f"Warnings: {result.stderr.strip()}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {md_path}: {e.stderr}", file=sys.stderr)
        raise

def main():
    """Main entry point."""
    docs_dir = Path(__file__).parent
    md_files = list(docs_dir.glob("*.md"))

    if not md_files:
        print("No Markdown files found in docs/ directory.")
        return

    print(f"Found {len(md_files)} Markdown files.")

    converted = 0
    skipped = 0
    errors = 0

    for md_path in md_files:
        html_path = md_path.with_suffix('.html')

        if should_regenerate(md_path, html_path):
            print(f"Converting {md_path.name} -> {html_path.name}...")
            try:
                convert_markdown_to_html(md_path, html_path)
                converted += 1
            except Exception as e:
                print(f"Failed to convert {md_path.name}: {e}")
                errors += 1
        else:
            print(f"Skipping {md_path.name} (HTML is up-to-date).")
            skipped += 1

    print("\n=== Summary ===")
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    if errors > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()