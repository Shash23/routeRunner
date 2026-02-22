#!/usr/bin/env python3
"""Run RolledBadger: load-response model + workout interpreter."""
import os
import sys
from pathlib import Path

# Run from project root so core-model imports work
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "core-model"))
os.chdir(project_root)

import load_response_model

if __name__ == "__main__":
    load_response_model.main()
