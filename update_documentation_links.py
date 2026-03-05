#!/usr/bin/env python3
"""Update documentation links to include new visualizations."""

from pathlib import Path
import re

def update_html_links(html_file, new_links):
    """Update HTML file with new navigation links."""
    try:
        with open(html_file, 'r') as f:
            content = f.read()

        # Find the navigation section or create one
        nav_pattern = r'(<div[^>]*class="[^"]*nav[^"]*"[^>]*>.*?</div>)'
        nav_match = re.search(nav_pattern, content, re.DOTALL | re.IGNORECASE)

        if nav_match:
            # Update existing navigation
            nav_content = nav_match.group(1)
            new_nav = nav_content
            for text, url in new_links:
                if text not in nav_content and url not in nav_content:
                    # Add new link
                    link_pattern = r'(</a>\s*</div>)'
                    new_link = f'<a href="{url}">{text}</a>\n      \\1'
                    new_nav = re.sub(link_pattern, new_link, new_nav, flags=re.DOTALL)

            content = content.replace(nav_content, new_nav)
        else:
            # Create new navigation section before closing body
            nav_html = '\n    <div class="container mt-4">\n      <div class="alert alert-info">\n        <h5>Related Reports</h5>\n        <p>'
            for text, url in new_links:
                nav_html += f'<a href="{url}" class="btn btn-outline-primary btn-sm me-2">{text}</a> '
            nav_html += '</p>\n      </div>\n    </div>\n'

            body_close_pattern = r'(</body>)'
            content = re.sub(body_close_pattern, nav_html + '\\1', content)

        with open(html_file, 'w') as f:
            f.write(content)

        print(f"  ✓ Updated: {html_file}")

    except Exception as e:
        print(f"  ⚠️  Error updating {html_file}: {e}")

def main():
    """Main function."""
    print("=" * 70)
    print("UPDATING DOCUMENTATION LINKS")
    print("=" * 70)

    # Define all reports and their relationships
    reports = {
        "enhanced_sensitivity_report/index.html": "Enhanced Sensitivity Report",
        "enhanced_model/index.html": "Enhanced Model Overview",
        "enhanced_model_visualizations/index.html": "Enhanced Model Visualizations",
        "missing_visualizations/index.html": "Missing Visualizations",
    }

    # Common navigation links for all reports
    common_links = [
        ("Enhanced Model", "../enhanced_model/index.html"),
        ("Sensitivity Report", "../enhanced_sensitivity_report/index.html"),
        ("Visualizations", "../enhanced_model_visualizations/index.html"),
        ("Time Series", "../missing_visualizations/index.html"),
    ]

    # Update each HTML file
    for html_path, description in reports.items():
        html_file = Path("docs") / html_path
        if html_file.exists():
            print(f"\nUpdating {description}:")
            update_html_links(html_file, common_links)
        else:
            print(f"\n❌ File not found: {html_file}")

    # Create a master index page
    print("\nCreating master index page...")
    master_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Sensitivity Model: Complete Analysis Suite</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            padding-top: 20px;
            padding-bottom: 40px;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 3rem 0;
            margin-bottom: 2rem;
            border-radius: 0 0 20px 20px;
        }
        .report-card {
            background: white;
            border-radius: 10px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid #dee2e6;
            transition: transform 0.2s;
        }
        .report-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .btn-report {
            padding: 0.75rem 1.5rem;
            font-size: 1.1rem;
        }
    </style>
    <!-- Simple Analytics -->
    <script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1 class="display-4">Enhanced Sensitivity Model</h1>
            <p class="lead">Complete Analysis Suite with Conservative Sampling</p>
            <p class="mb-0">adapt_delta=0.99 | max_treedepth=12 | 4 chains × 2000 iterations</p>
        </div>

        <div class="alert alert-success">
            <h4>🎯 Comprehensive Analysis Suite</h4>
            <p>This suite provides complete visualization and analysis of the enhanced sensitivity model with conservative sampling for reliable inference.</p>
        </div>

        <div class="row">
            <div class="col-md-6">
                <div class="report-card">
                    <h3>📊 Enhanced Sensitivity Report</h3>
                    <p>Detailed analysis report with model diagnostics, parameter estimates, and performance metrics.</p>
                    <ul>
                        <li>Model diagnostics and convergence</li>
                        <li>Parameter estimates with credible intervals</li>
                        <li>Performance metrics and comparisons</li>
                        <li>Technical implementation details</li>
                    </ul>
                    <a href="enhanced_sensitivity_report/index.html" class="btn btn-primary btn-report">View Report</a>
                </div>
            </div>
            <div class="col-md-6">
                <div class="report-card">
                    <h3>📈 Enhanced Model Visualizations</h3>
                    <p>Comprehensive visualizations including component time series, convergence diagnostics, and parameter distributions.</p>
                    <ul>
                        <li>Component time series analysis</li>
                        <li>Convergence diagnostics (R-hat, ESS)</li>
                        <li>Parameter distribution forest plots</li>
                        <li>Fitness state evolution</li>
                    </ul>
                    <a href="enhanced_model_visualizations/index.html" class="btn btn-primary btn-report">View Visualizations</a>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-6">
                <div class="report-card">
                    <h3>⏱️ Missing Visualizations</h3>
                    <p>Key time series and component analysis showing weight predictions, component breakdown, and fitness state evolution.</p>
                    <ul>
                        <li>Weight predictions vs observations</li>
                        <li>Component breakdown over time</li>
                        <li>Fitness state evolution with activity overlay</li>
                        <li>Variance proportions and half-lives</li>
                    </ul>
                    <a href="missing_visualizations/index.html" class="btn btn-primary btn-report">View Time Series</a>
                </div>
            </div>
            <div class="col-md-6">
                <div class="report-card">
                    <h3>🔬 Enhanced Model Overview</h3>
                    <p>Model overview with technical specifications, sampling configuration, and implementation details.</p>
                    <ul>
                        <li>Model specification and equations</li>
                        <li>Sampling configuration</li>
                        <li>Implementation details</li>
                        <li>Technical specifications</li>
                    </ul>
                    <a href="enhanced_model/index.html" class="btn btn-primary btn-report">View Overview</a>
                </div>
            </div>
        </div>

        <div class="alert alert-info">
            <h5>📋 Model Configuration</h5>
            <table class="table table-sm">
                <tr><th>Parameter</th><th>Value</th><th>Purpose</th></tr>
                <tr><td>adapt_delta</td><td>0.99</td><td>Conservative sampling, reduces divergent transitions</td></tr>
                <tr><td>max_treedepth</td><td>12</td><td>Deeper exploration of posterior</td></tr>
                <tr><td>warmup iterations</td><td>1000</td><td>Extended warmup for better adaptation</td></tr>
                <tr><td>sampling iterations</td><td>1000</td><td>Sufficient samples for reliable inference</td></tr>
                <tr><td>chains</td><td>4</td><td>Multiple chains for convergence diagnostics</td></tr>
                <tr><td>prediction points</td><td>200</td><td>Dense grid for smooth predictions</td></tr>
                <tr><td>fitness components</td><td>4</td><td>Aerobic/Strength × Short/Long-term</td></tr>
            </table>
        </div>

        <div class="text-center text-muted mt-4">
            <hr>
            <p>Enhanced Sensitivity Model Analysis Suite | Generated: March 2026</p>
            <p>All reports include Simple Analytics tracking for usage monitoring</p>
        </div>
    </div>
</body>
</html>'''

    master_file = Path("docs") / "enhanced_model_suite" / "index.html"
    master_file.parent.mkdir(exist_ok=True)

    with open(master_file, 'w') as f:
        f.write(master_content)

    print(f"  ✓ Created master index: {master_file}")

    print("\n" + "=" * 70)
    print("✅ DOCUMENTATION LINKS UPDATED SUCCESSFULLY")
    print("=" * 70)
    print("\nMaster index page: docs/enhanced_model_suite/index.html")
    print("\nIndividual Reports:")
    for html_path, description in reports.items():
        print(f"  - {description}: docs/{html_path}")

if __name__ == "__main__":
    main()