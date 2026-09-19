#!/usr/bin/env python3
import os
import sys

# Add 2.src to sys.path and execute 2.src/train_test_5omics.py
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2.src')
sys.path.insert(0, src_dir)

from train_test_5omics import parse_args, main

if __name__ == '__main__':
    args = parse_args()
    main(args)
