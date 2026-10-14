#!/bin/sh
# 修改 main.py 或 data/ 後, 同步到 docs/ (GitHub Pages stlite Demo)
set -e
cd "$(dirname "$0")"
cp main.py docs/main.py
cp data/50_Startups.csv data/BostonHousing.csv docs/
echo "docs/ synced"
