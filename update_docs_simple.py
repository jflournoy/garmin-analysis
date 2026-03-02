#!/usr/bin/env python3
"""Simple script to update documentation for enhanced sensitivity model."""

from pathlib import Path
from datetime import datetime
import shutil

PROJECT_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT / "docs"
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

def update_main_index():
    """Update main docs/index.html."""
    index_path = DOCS_DIR / "index.html"

    with open(index_path, 'r') as f:
        content = f.read()

    # Update the featured card
    old_featured = '''<div class="doc-card">
            <h3>Four-Fitness State-Space Analysis</h3>
            <p>Comprehensive analysis of the four-fitness state-space model with baseline fitness equilibrium at 0. Shows how aerobic and strength training affect weight over short-term and long-term time scales with interactive visualizations.</p>
            <a href="four_fitness_comprehensive/index.html">View Primary Analysis →</a>
        </div>'''

    new_featured = f'''<div class="doc-card">
            <h3>🌟 Enhanced Sensitivity Model (Primary)</h3>
            <p><strong>Featured Model:</strong> Comprehensive analysis of enhanced sensitivity model with conservative sampling (adapt_delta=0.99) and smooth continuous predictions. This is now the primary model for all analysis.</p>
            <a href="enhanced_sensitivity_report/index.html">View Primary Analysis →</a>
        </div>'''

    content = content.replace(old_featured, new_featured)

    # Update footer date
    content = content.replace(
        'Documentation generated: 2026-03-02',
        f'Documentation generated: {CURRENT_DATE}'
    )

    # Add notice about enhanced model
    notice = f'''<div class="notice-box">
            <strong>🎯 Enhanced Sensitivity Model Now Primary:</strong> All documentation has been updated to feature the enhanced sensitivity model with conservative sampling (adapt_delta=0.99, max_treedepth=12). This model provides reliable inference with proper uncertainty quantification and smooth continuous predictions.
        </div>'''

    # Insert after header
    header_end = content.find('</header>') + len('</header>')
    content = content[:header_end] + '\n\n' + notice + content[header_end:]

    with open(index_path, 'w') as f:
        f.write(content)

    print("✓ Updated main index.html")

def create_enhanced_landing_page():
    """Create enhanced model landing page."""
    landing_dir = DOCS_DIR / "enhanced_model"
    landing_dir.mkdir(exist_ok=True)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Sensitivity Model: Primary Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            padding-top: 20px;
            padding-bottom: 40px;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 3rem 0;
            margin-bottom: 2rem;
            border-radius: 0 0 20px 20px;
        }}
        .feature-card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #3498db;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section {{
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
    </style>
    <!-- Simple Analytics -->
    <script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
</head>
<body>
    <div class="container">
        <div class="header text-center">
            <h1 class="display-4">Enhanced Sensitivity Model</h1>
            <p class="lead">Primary Model with Conservative Sampling and Smooth Predictions</p>
            <p class="mb-0">Generated: {CURRENT_DATE}</p>
        </div>

        <div class="alert alert-success">
            <h4>🎯 Primary Model Status</h4>
            <p>This enhanced sensitivity model is now the primary model for all analysis. It features conservative sampling (adapt_delta=0.99) for reliable inference and smooth continuous predictions.</p>
        </div>

        <div class="row">
            <div class="col-md-4">
                <div class="feature-card">
                    <h5>📊 Comprehensive Report</h5>
                    <p>Detailed analysis with parameter estimates, variance decomposition, and convergence diagnostics.</p>
                    <a href="../enhanced_sensitivity_report/index.html" class="btn btn-primary btn-sm">View Report</a>
                </div>
            </div>
            <div class="col-md-4">
                <div class="feature-card">
                    <h5>🔬 Technical Details</h5>
                    <p>Mathematical foundations, Stan implementation, and sampling configuration.</p>
                    <a href="#technical" class="btn btn-primary btn-sm">View Details</a>
                </div>
            </div>
            <div class="col-md-4">
                <div class="feature-card">
                    <h5>📈 Predictions</h5>
                    <p>Smooth continuous predictions with 95% credible intervals and component breakdown.</p>
                    <a href="../enhanced_sensitivity_report/index.html#predictions" class="btn btn-primary btn-sm">View Predictions</a>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Key Features</h2>
            <ul>
                <li><strong>Conservative Sampling:</strong> adapt_delta=0.99 reduces divergent transitions</li>
                <li><strong>Deep Exploration:</strong> max_treedepth=12 for complex posterior landscapes</li>
                <li><strong>Smooth Predictions:</strong> 200-point dense grid for continuous curves</li>
                <li><strong>Excellent Convergence:</strong> All parameters R-hat ≤ 1.1</li>
                <li><strong>Component Breakdown:</strong> Visual decomposition of model contributions</li>
                <li><strong>Uncertainty Quantification:</strong> Proper credible intervals from conservative sampling</li>
            </ul>
        </div>

        <div class="section" id="technical">
            <h2>Technical Specifications</h2>
            <table class="table table-striped">
                <tr><th>Parameter</th><th>Value</th><th>Purpose</th></tr>
                <tr><td>adapt_delta</td><td>0.99</td><td>Conservative sampling, reduces divergent transitions</td></tr>
                <tr><td>max_treedepth</td><td>12</td><td>Deeper exploration of posterior</td></tr>
                <tr><td>warmup iterations</td><td>1000</td><td>Extended warmup for better adaptation</td></tr>
                <tr><td>sampling iterations</td><td>1000</td><td>Sufficient samples for reliable inference</td></tr>
                <tr><td>chains</td><td>4</td><td>Multiple chains for convergence diagnostics</td></tr>
                <tr><td>prediction points</td><td>200</td><td>Dense grid for smooth predictions</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>Key Findings</h2>
            <div class="alert alert-info">
                <h5>GP Constraint Test</h5>
                <p>alpha_gp = 0.7601 exceeds the 0.5 constraint, indicating the Gaussian Process needs more flexibility to capture trends.</p>
            </div>
            <div class="alert alert-info">
                <h5>Weight Effects</h5>
                <p>Expected physiological patterns detected: negative aerobic effects, positive long-term strength effects.</p>
            </div>
            <div class="alert alert-info">
                <h5>Variance Decomposition</h5>
                <p>83% of variance from GP trends, ~1% from fitness effects, 2.8% from daily cycles.</p>
            </div>
        </div>

        <div class="section">
            <h2>Running the Model</h2>
            <pre><code># Run enhanced sensitivity model
uv run python run_enhanced_sensitivity.py

# Check results
uv run python check_enhanced_model.py

# Generate report
uv run python create_enhanced_sensitivity_report.py</code></pre>
        </div>

        <div class="text-center text-muted">
            <hr>
            <p>Enhanced Sensitivity Model: Primary Analysis</p>
            <p>Generated on {CURRENT_DATE} | Model: weight_state_space_four_fitness_sensitivity_pred.stan</p>
        </div>
    </div>
</body>
</html>'''

    output_file = landing_dir / "index.html"
    with open(output_file, 'w') as f:
        f.write(html_content)

    print("✓ Created enhanced model landing page")

def update_existing_docs():
    """Add notices to existing documentation."""
    docs_to_update = [
        DOCS_DIR / "four_fitness_comprehensive" / "index.html",
        DOCS_DIR / "four_fitness_educational" / "index.html",
        DOCS_DIR / "weight_fitness_report" / "index.html"
    ]

    notice = f'''<div class="alert alert-info mt-3 mb-4">
        <strong>🎯 Note:</strong> This documentation references previous models. For the latest enhanced sensitivity model with conservative sampling (adapt_delta=0.99) and smooth predictions, see the <a href="../enhanced_model/index.html">Enhanced Sensitivity Model</a>.
    </div>'''

    for doc_path in docs_to_update:
        if doc_path.exists():
            with open(doc_path, 'r') as f:
                content = f.read()

            # Insert notice after header
            header_end = content.find('</header>')
            if header_end != -1:
                header_end += len('</header>')
                content = content[:header_end] + '\n\n' + notice + content[header_end:]

            with open(doc_path, 'w') as f:
                f.write(content)

            print(f"✓ Updated {doc_path.name}")

def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("UPDATING DOCUMENTATION FOR ENHANCED SENSITIVITY MODEL")
    print("=" * 70)

    update_main_index()
    create_enhanced_landing_page()
    update_existing_docs()

    print("\n" + "=" * 70)
    print("DOCUMENTATION UPDATE COMPLETE")
    print("=" * 70)
    print("\nEnhanced model is now featured as primary in:")
    print("1. Main documentation index (docs/index.html)")
    print("2. New landing page (docs/enhanced_model/index.html)")
    print("3. All existing documentation (with update notices)")
    print(f"\nOpen docs/enhanced_model/index.html to view the primary model documentation.")

if __name__ == "__main__":
    main()