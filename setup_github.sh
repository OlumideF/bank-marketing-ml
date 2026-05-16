#!/bin/bash
# =============================================================================
# setup_github.sh — Initialize the repo and push to GitHub
# =============================================================================
# Usage:
#   chmod +x setup_github.sh
#   ./setup_github.sh YOUR_GITHUB_USERNAME bank-marketing-ml
# =============================================================================

set -e

GITHUB_USER="${1:-YOUR_USERNAME}"
REPO_NAME="${2:-bank-marketing-ml}"

echo "======================================"
echo "  Bank Marketing ML — GitHub Setup"
echo "======================================"
echo ""
echo "  GitHub user : $GITHUB_USER"
echo "  Repo name   : $REPO_NAME"
echo ""

# Init git
git init
git add .
git commit -m "Initial commit: UCI Bank Marketing ML pipeline

- XGBoost classifier with SMOTE for class imbalance (89/11 split)
- Two model variants: with/without duration (data leakage documentation)
- SHAP analysis with economic interpretation of euribor3m
- Lift curve showing 3.9x improvement at top 20% of customers
- Full EDA and evaluation dashboard
- Auto-downloads UCI dataset on first run
"

echo ""
echo "======================================"
echo "  Next steps:"
echo "======================================"
echo ""
echo "  1. Create the repo on GitHub (do this in your browser):"
echo "     https://github.com/new"
echo "     Name: $REPO_NAME"
echo "     Visibility: Public"
echo "     Do NOT initialize with README (we have one already)"
echo ""
echo "  2. Then run these commands:"
echo ""
echo "     git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
echo "     git branch -M main"
echo "     git push -u origin main"
echo ""
echo "  3. Optional — run the pipeline to generate output plots:"
echo "     pip install -r requirements.txt"
echo "     python src/pipeline.py"
echo "     git add outputs/"
echo "     git commit -m 'Add generated output plots'"
echo "     git push"
echo ""
echo "  Done! Your repo will be live at:"
echo "  https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
