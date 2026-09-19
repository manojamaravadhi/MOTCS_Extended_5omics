#!/usr/bin/env python3
import os
import sys

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2.src')
sys.path.insert(0, src_dir)

from run_ablation_benchmark import main

if __name__ == '__main__':
    main()
