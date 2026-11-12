#!/bin/bash
# package.sh - Create submission zip
# Usage: bash scripts/package.sh [GROUP_NUMBER] [TITLE]

set -e

GROUP="${1:-GroupXX}"
TITLE="${2:-Farmer_Advisory_Chatbot_Multilingual_NLP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if command -v python3 &>/dev/null; then
    python3 scripts/package.py "$GROUP" "$TITLE"
elif command -v python &>/dev/null; then
    python scripts/package.py "$GROUP" "$TITLE"
else
    ZIPNAME="${GROUP}_code_${TITLE}.zip"
    echo "Packaging via standard zip utility: $ZIPNAME"
    zip -r "$ZIPNAME" . \
        -x "data/raw/*" \
        -x "data/plantvillage/*" \
        -x "data/plantdoc/*" \
        -x "checkpoints/*.pth" \
        -x "checkpoints/**/*.pth" \
        -x "venv/*" \
        -x ".venv/*" \
        -x "__pycache__/*" \
        -x "**/__pycache__/*" \
        -x "*.pyc" \
        -x ".git/*" \
        -x "**/.git/*" \
        -x "*.zip"
fi
