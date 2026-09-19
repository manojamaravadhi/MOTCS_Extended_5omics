#!/usr/bin/env python
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
src_script = os.path.join(cur_dir, "2.src", "arrange_all_cancer_cohorts.py")
if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, src_script] + sys.argv[1:])
