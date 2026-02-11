"""
Wrapper module for generate-docs entry point.
This allows the entry point to work with the editable installation.
"""
import sys
import os

# Add the project root to sys.path so docs module can be imported
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from docs.generate_docs import main

if __name__ == "__main__":
    sys.exit(main())