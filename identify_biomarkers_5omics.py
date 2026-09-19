#!/usr/bin/env python3
import os
import sys

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2.src')
sys.path.insert(0, src_dir)

from identify_biomarkers_5omics import parse_args, main

if __name__ == '__main__':
    args = parse_args()
    main(args)
