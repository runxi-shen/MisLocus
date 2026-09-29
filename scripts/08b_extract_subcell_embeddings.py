#!/usr/bin/env python3
"""Extract frozen four-channel SubCell embeddings after 08a preflight; no training."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor/subcell_embed')]

from prot_loc_benchmark.representations.subcell_extract import main

if __name__ == '__main__':
    main(frozen=True)
