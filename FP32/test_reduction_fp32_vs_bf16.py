#!/usr/bin/env python3
"""Latency + deterministic + batch_invariance: only post-matmul reduction in FP32 vs BF16. Entry point."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FP32.run_tests import main

if __name__ == "__main__":
    main()
