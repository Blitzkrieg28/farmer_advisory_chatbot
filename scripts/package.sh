#!/bin/bash
# package.sh - Create submission zip
# Usage: bash scripts/package.sh [GROUP_NUMBER] [TITLE]
#
# Creates: GroupXX_code_Title.zip
# Excludes: raw data, large checkpoints, virtual environments, __pycache__

set -e

GROUP="${1:-GroupXX}"
TITLE="${2:-Farmer_Advisory_Chatbot_Multilingual_NLP}"
ZIPNAME="${GROUP}_code_${TITLE}.zip"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Packaging project: $ZIPNAME"
echo "From directory: $PROJECT_DIR"

# Create zip excluding large/generated files
zip -r "$ZIPNAME" . \
    -x "data/raw/*" \
    -x "data/plantvillage/*" \
    -x "data/plantdoc/*" \
    -x "checkpoints/*.pth" \
    -x "checkpoints/**/*.pth" \
    -x "checkpoints/retriever/best_model/*" \
    -x "venv/*" \
    -x ".venv/*" \
    -x "__pycache__/*" \
    -x "**/__pycache__/*" \
    -x "*.pyc" \
    -x ".git/*" \
    -x "**/.git/*" \
    -x ".ipynb_checkpoints/*" \
    -x "**/.ipynb_checkpoints/*" \
    -x "*.zip" \
    -x "logs/*.log" \
    -x "wandb/*"

echo ""
echo "Created: $ZIPNAME"
echo "Size: $(du -h "$ZIPNAME" | cut -f1)"
echo ""
echo "Included:"
echo "  - All source code (.py, .yaml, .json, .md, .sh)"
echo "  - Config files"
echo "  - Knowledge base template"
echo "  - Report assets (figures, tables)"
echo "  - Notebooks"
echo "  - CSV logs and metrics"
echo "  - requirements.txt, README.md"
echo ""
echo "Excluded:"
echo "  - Raw image datasets"
echo "  - Large model checkpoints (.pth)"
echo "  - Virtual environments"
echo "  - __pycache__ and .pyc files"

