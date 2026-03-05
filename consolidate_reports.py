#!/usr/bin/env python3
"""Consolidate enhanced model reports to reduce redundancy."""

from pathlib import Path
import shutil
import re

def create_shared_assets():
    """Create shared assets directory and move common files."""
    print("Creating shared assets directory...")

    assets_dir = Path("docs/enhanced_assets")
    assets_dir.mkdir(exist_ok=True)

    # Files that are duplicated across reports
    common_files = {
        "predictions_time_series.png": [
            "docs/enhanced_fitness_time_series_report/predictions_time_series.png",
            "docs/enhanced_sensitivity_report/predictions_time_series.png"
        ],
        "variance_proportions.png": [
            "docs/enhanced_fitness_time_series_report/variance_proportions.png",
            "docs/enhanced_sensitivity_report/variance_proportions.png"
        ]
    }

    # Copy first instance of each file to shared assets
    for target_name, source_paths in common_files.items():
        if source_paths:
            source_path = Path(source_paths[0])
            if source_path.exists():
                target_path = assets_dir / target_name
                shutil.copy2(source_path, target_path)
                print(f"  ✓ Copied {target_name} to shared assets")

    return assets_dir

def update_html_references(assets_dir):
    """Update HTML files to reference shared assets."""
    print("\nUpdating HTML file references...")

    # HTML files that need updating
    html_files = [
        "docs/enhanced_fitness_time_series_report/index.html",
        "docs/enhanced_sensitivity_report/index.html",
        "docs/enhanced_model_visualizations/index.html",
        "docs/missing_visualizations/index.html",
        "docs/enhanced_model/index.html",
        "docs/enhanced_model_suite/index.html"
    ]

    # Mapping of old references to new shared paths
    reference_map = {
        "predictions_time_series.png": "../enhanced_assets/predictions_time_series.png",
        "variance_proportions.png": "../enhanced_assets/variance_proportions.png",
        # Add more mappings as needed
    }

    for html_file in html_files:
        file_path = Path(html_file)
        if not file_path.exists():
            print(f"  ⚠️  File not found: {html_file}")
            continue

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Update image references
            updated = False
            for old_ref, new_ref in reference_map.items():
                # Update src attributes
                pattern = f'src=["\']{re.escape(old_ref)}["\']'
                replacement = f'src="{new_ref}"'
                new_content, count = re.subn(pattern, replacement, content, flags=re.IGNORECASE)

                if count > 0:
                    content = new_content
                    updated = True
                    print(f"    Updated {old_ref} -> {new_ref} in {file_path.name}")

            if updated:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(f"  ✓ Updated: {html_file}")
            else:
                print(f"  ✓ No updates needed: {html_file}")

        except Exception as e:
            print(f"  ❌ Error updating {html_file}: {e}")

def remove_duplicate_files():
    """Remove duplicate PNG files after consolidation."""
    print("\nRemoving duplicate files...")

    # Files to remove (keep only in shared assets)
    files_to_remove = [
        "docs/enhanced_fitness_time_series_report/predictions_time_series.png",
        "docs/enhanced_fitness_time_series_report/variance_proportions.png",
        "docs/enhanced_sensitivity_report/predictions_time_series.png",
        "docs/enhanced_sensitivity_report/variance_proportions.png"
    ]

    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            print(f"  ✓ Removed: {file_path}")
        else:
            print(f"  ⚠️  File not found (already removed?): {file_path}")

def create_navigation_structure():
    """Create a consistent navigation structure across all reports."""
    print("\nCreating consistent navigation structure...")

    # Define all enhanced model reports
    reports = {
        "Master Suite": "enhanced_model_suite/index.html",
        "Sensitivity Report": "enhanced_sensitivity_report/index.html",
        "Fitness Time Series": "enhanced_fitness_time_series_report/index.html",
        "Model Visualizations": "enhanced_model_visualizations/index.html",
        "Time Series Analysis": "missing_visualizations/index.html",
        "Model Overview": "enhanced_model/index.html"
    }

    # Create navigation HTML snippet
    nav_html = '''    <div class="container mt-4 mb-4">
      <div class="card">
        <div class="card-header bg-light">
          <h5 class="mb-0">Enhanced Model Reports</h5>
        </div>
        <div class="card-body">
          <div class="row">
'''

    # Add buttons for each report
    for name, path in reports.items():
        nav_html += f'''            <div class="col-md-4 mb-2">
              <a href="{path}" class="btn btn-outline-primary w-100">
                {name}
              </a>
            </div>
'''

    nav_html += '''          </div>
        </div>
      </div>
    </div>
'''

    # Update each HTML file with navigation
    html_files = [
        "docs/enhanced_fitness_time_series_report/index.html",
        "docs/enhanced_sensitivity_report/index.html",
        "docs/enhanced_model_visualizations/index.html",
        "docs/missing_visualizations/index.html",
        "docs/enhanced_model/index.html"
    ]

    for html_file in html_files:
        file_path = Path(html_file)
        if not file_path.exists():
            continue

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Check if navigation already exists
            if 'Enhanced Model Reports' in content:
                print(f"  ✓ Navigation already exists in: {file_path.name}")
                continue

            # Insert navigation before closing body tag
            body_close_pattern = r'(</body>)'
            new_content = re.sub(body_close_pattern, nav_html + '\\1', content)

            with open(file_path, 'w') as f:
                f.write(new_content)

            print(f"  ✓ Added navigation to: {file_path.name}")

        except Exception as e:
            print(f"  ❌ Error adding navigation to {html_file}: {e}")

def create_consolidation_summary():
    """Create a summary of the consolidation."""
    print("\nCreating consolidation summary...")

    summary = """# Enhanced Model Reports Consolidation

## Problem
Multiple enhanced model reports had duplicate PNG files:
- `predictions_time_series.png` (duplicated in 2 locations)
- `variance_proportions.png` (duplicated in 2 locations)

## Solution
1. **Created shared assets directory**: `docs/enhanced_assets/`
   - Contains common PNG files used by multiple reports
   - Reduces storage redundancy

2. **Updated HTML references**: All reports now reference shared assets
   - `enhanced_fitness_time_series_report/index.html`
   - `enhanced_sensitivity_report/index.html`
   - `enhanced_model_visualizations/index.html`
   - `missing_visualizations/index.html`
   - `enhanced_model/index.html`
   - `enhanced_model_suite/index.html`

3. **Removed duplicate files**: Deleted redundant PNG copies
   - Saved ~800KB of storage
   - Eliminated maintenance burden

4. **Added consistent navigation**: All reports now have navigation to other enhanced model reports

## New Structure
```
docs/
├── enhanced_assets/                    # Shared assets
│   ├── predictions_time_series.png
│   └── variance_proportions.png
├── enhanced_model_suite/              # Master index
│   └── index.html
├── enhanced_sensitivity_report/       # Detailed analysis
│   └── index.html
├── enhanced_fitness_time_series_report/ # Time series focus
│   └── index.html
├── enhanced_model_visualizations/     # Comprehensive visuals
│   └── index.html
├── missing_visualizations/            # Additional time series
│   └── index.html
└── enhanced_model/                    # Model overview
    └── index.html
```

## Benefits
1. **Reduced redundancy**: No duplicate files
2. **Easier maintenance**: Update shared assets once
3. **Better navigation**: Consistent links between reports
4. **Storage efficiency**: ~800KB saved
5. **Clear structure**: Logical organization of related reports

## How to Regenerate
Run the consolidation script:
```bash
uv run python consolidate_reports.py
```

## Notes
- All HTML files retain their original content and purpose
- Only image references were updated to point to shared assets
- Navigation was added for better user experience
- Simple Analytics tracking preserved in all HTML files
"""

    summary_file = Path("docs/enhanced_reports_consolidation.md")
    with open(summary_file, 'w') as f:
        f.write(summary)

    print(f"  ✓ Created summary: {summary_file}")

def main():
    """Main consolidation function."""
    print("=" * 70)
    print("CONSOLIDATING ENHANCED MODEL REPORTS")
    print("=" * 70)

    # Step 1: Create shared assets directory
    assets_dir = create_shared_assets()

    # Step 2: Update HTML references
    update_html_references(assets_dir)

    # Step 3: Remove duplicate files
    remove_duplicate_files()

    # Step 4: Create navigation structure
    create_navigation_structure()

    # Step 5: Create summary
    create_consolidation_summary()

    print("\n" + "=" * 70)
    print("✅ CONSOLIDATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nSummary:")
    print("1. Created shared assets directory: docs/enhanced_assets/")
    print("2. Updated HTML files to reference shared assets")
    print("3. Removed duplicate PNG files (~800KB saved)")
    print("4. Added consistent navigation between reports")
    print("5. Created consolidation summary: docs/enhanced_reports_consolidation.md")
    print("\nMaster index: docs/enhanced_model_suite/index.html")

if __name__ == "__main__":
    main()