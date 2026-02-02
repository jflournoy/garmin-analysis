#!/usr/bin/env python3
"""
Convert Markdown with MathJax to HTML.

This script converts a Markdown file to HTML with MathJax support
for rendering mathematical notation.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

try:
    import markdown
    from markdown.extensions.extra import ExtraExtension
    import pymdownx.arithmatex
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("  uv add markdown pymdown-extensions")
    sys.exit(1)

def convert_markdown_to_html(md_path: Path, html_path: Path = None):
    """Convert Markdown file to HTML with MathJax support."""

    if not md_path.exists():
        print(f"Error: Markdown file not found: {md_path}")
        sys.exit(1)

    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Configure markdown extensions for MathJax
    extensions = [
        ExtraExtension(),
        'pymdownx.arithmatex',
        'pymdownx.superfences',
        'pymdownx.tabbed',
        'markdown.extensions.tables',
        'markdown.extensions.codehilite',
    ]

    extension_configs = {
        'pymdownx.arithmatex': {
            'generic': True,  # Use generic mode for MathJax
        }
    }

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_content,
        extensions=extensions,
        extension_configs=extension_configs,
        output_format='html5'
    )

    # Create full HTML document with MathJax
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Bayesian Gaussian Process Model for Weight Analysis</title>

    <!-- MathJax Configuration -->
    <script>
    MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true,
            processEnvironments: true
        }},
        options: {{
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        }}
    }};
    </script>

    <!-- Load MathJax -->
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

    <!-- Basic styling -->
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}

        h1, h2, h3, h4 {{
            color: #2c3e50;
            margin-top: 1.5em;
            border-bottom: 1px solid #eee;
            padding-bottom: 0.3em;
        }}

        h1 {{ border-bottom: 2px solid #3498db; }}

        code {{
            background-color: #f8f8f8;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }}

        pre {{
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}

        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}

        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}

        tr:nth-child(even) {{ background-color: #f9f9f9; }}

        .math, .arithmatex {{
            overflow-x: auto;
            overflow-y: hidden;
        }}

        .mjx-chtml {{ outline: none; }}

        footer {{
            margin-top: 3em;
            padding-top: 1em;
            border-top: 1px solid #eee;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    {html_body}

    <footer>
        <p>Generated from: {md_path.name}</p>
        <p>Last updated: {datetime.fromtimestamp(os.path.getmtime(md_path)).strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Using MathJax v3 for mathematical notation rendering.</p>
    </footer>
</body>
</html>
"""

    # Determine output path
    if html_path is None:
        html_path = md_path.with_suffix('.html')

    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"✓ Converted {md_path} to {html_path}")
    print("✓ MathJax included for mathematical notation rendering")
    print(f"✓ Open {html_path} in a web browser to view the document")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_to_html.py <markdown_file.md> [output_file.html]")
        print("\nExample:")
        print("  python convert_to_html.py weight_gp_model_explanation.md")
        print("  python convert_to_html.py weight_gp_model_explanation.md model_explanation.html")
        sys.exit(1)

    md_path = Path(sys.argv[1])

    if len(sys.argv) > 2:
        html_path = Path(sys.argv[2])
    else:
        html_path = None

    try:
        convert_markdown_to_html(md_path, html_path)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()