#!/usr/bin/env python3
"""Plan or sequentially run a bounded analysis from pinned feature inputs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prot_loc_benchmark.workflows.runner import main

if __name__ == '__main__':
    main()
