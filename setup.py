"""
setup.py — Run this first to create the full project folder structure.

Usage:
    python setup.py
"""

import os
from pathlib import Path

BASE = Path(__file__).parent

dirs = [
    "data",
    "notebooks",
    "src",
    "static",
    "models/bert_intent",
    "tests",
]

for d in dirs:
    (BASE / d).mkdir(parents=True, exist_ok=True)
    print(f"  created  {d}/")

print("\nFolder structure ready. Now run the project:")
print("  pip install -r requirements.txt")
print("  python src/train_intent.py")
print("  uvicorn src.api:app --reload")
