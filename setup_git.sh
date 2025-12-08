#!/bin/bash

# Git Setup Script for African Commodities Paradox Project
# This script creates meaningful commits for the initial project setup
# Author: Abraham Adegoke
# Date: November 2025

echo "🚀 Setting up Git commits for African Commodities Paradox project"
echo "=================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to make a commit
make_commit() {
    local files="$1"
    local message="$2"
    local commit_num="$3"
    
    echo -e "${BLUE}Commit #${commit_num}:${NC} ${message}"
    git add $files
    git commit -m "$message"
    echo -e "${GREEN}✓ Done${NC}"
    echo ""
}

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Git not initialized. Run 'git init' first."
    exit 1
fi

echo "📝 Creating structured commits..."
echo ""

# Phase 1: Project Setup & Configuration
echo "=== Phase 1: Project Setup & Configuration ==="
echo ""

make_commit ".gitignore" \
    "chore: add .gitignore for Python project" \
    "1"

make_commit "README.md PROPOSAL.md" \
    "docs: add project README and research proposal" \
    "2"

make_commit "requirements.txt" \
    "feat: add Python dependencies for data science and ML stack" \
    "3"

make_commit "configs/countries.yaml" \
    "feat: add country configuration with African countries subsets" \
    "4"

# Phase 2: Data Collection Module
echo "=== Phase 2: Data Collection Module ==="
echo ""

make_commit "src/__init__.py src/data_io/__init__.py" \
    "feat: initialize src package structure" \
    "5"

make_commit "src/data_io/worldbank.py" \
    "feat: implement World Bank WDI API client with CDI calculation" \
    "6"

make_commit "scripts/download_data.py" \
    "feat: add data download script with flexible country selection" \
    "7"

# Phase 3: Analysis Pipeline
echo "=== Phase 3: Analysis Pipeline ==="
echo ""

make_commit "scripts/run_analysis.py" \
    "feat: implement complete analysis pipeline with feature engineering" \
    "8"

# Phase 4: Interactive Notebooks
echo "=== Phase 4: Interactive Notebooks ==="
echo ""

make_commit "notebooks/00_quickstart.ipynb" \
    "docs: add interactive quickstart notebook for exploratory analysis" \
    "9"

# Phase 5: Data directories (if they exist and have files)
echo "=== Phase 5: Data Structure ==="
echo ""

if [ -f "data/raw/.gitkeep" ] || [ -f "data/raw/worldbank_wdi.csv" ]; then
    make_commit "data/raw/" \
        "data: add raw data directory structure" \
        "10"
fi

if [ -f "data/processed/.gitkeep" ] || [ -f "data/processed/features_ready.csv" ]; then
    make_commit "data/processed/" \
        "data: add processed data directory with feature-engineered datasets" \
        "11"
fi

# Summary
echo "=================================================================="
echo -e "${GREEN}✅ Git commits completed successfully!${NC}"
echo ""
echo "📊 Summary:"
git log --oneline | head -15
echo ""
echo "💡 Next steps:"
echo "  1. Push to GitHub: git push origin main"
echo "  2. Continue development with frequent commits"
echo "  3. Aim for 20+ meaningful commits total"
echo ""
echo "🎯 Current commit count: $(git log --oneline | wc -l)"