#!/usr/bin/env python3
"""
Auto-Git-Sync Daemon
====================
Monitors the repository for any code, data, or result changes.
Automatically stages, commits, and pushes to GitHub in real-time.

Usage:
  # Start monitoring in foreground
  python auto_git_sync.py

  # Start monitoring in background (daemon mode)
  nohup python auto_git_sync.py --interval 30 > auto_git_sync.log 2>&1 &
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, cwd=REPO_DIR):
    res = subprocess.run(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def check_and_sync(branch="main"):
    # Check if there are unstaged, staged, or untracked changes
    rc, status_out, _ = run_cmd("git status --porcelain")
    if not status_out:
        return False

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now_str}] 🔔 Changes detected:")
    lines = status_out.splitlines()
    for l in lines[:10]:
        print(f"   {l}")
    if len(lines) > 10:
        print(f"   ... and {len(lines)-10} more files")

    # 1. Stage changes
    print(f"[{now_str}] 📦 Staging changes (git add .)...")
    rc_add, _, err_add = run_cmd("git add .")
    if rc_add != 0:
        print(f"   ✗ Error during git add: {err_add}")
        return False

    # 2. Commit changes (no tags, no colons in commit message)
    time_stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    summary = f"Update {len(lines)} file(s) ({time_stamp})"
    print(f"[{now_str}] 📝 Committing: '{summary}'...")
    rc_commit, out_commit, err_commit = run_cmd(f'git commit -m "{summary}"')
    if rc_commit != 0:
        print(f"   ✗ Error during git commit: {err_commit}")
        return False

    # 3. Push changes
    print(f"[{now_str}] 🚀 Pushing to GitHub (git push origin {branch})...")
    rc_push, out_push, err_push = run_cmd(f"git push origin {branch}")
    if rc_push == 0:
        print(f"[{now_str}] ✓ Successfully pushed to GitHub!")
        return True
    else:
        print(f"   ⚠ Push failed or remote not reachable yet: {err_push or out_push}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Auto Git Sync Daemon")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds (default: 30)")
    parser.add_argument("--branch", type=str, default="main", help="Target git branch (default: main)")
    args = parser.parse_args()

    print("=" * 65)
    print("  AUTOMATIC GIT SYNC DAEMON ACTIVE")
    print("=" * 65)
    print(f"  Repository : {REPO_DIR}")
    print(f"  Target Branch : {args.branch}")
    print(f"  Poll Interval : Every {args.interval} seconds")
    print("  Monitoring for changes... (Press Ctrl+C to stop)")
    print("=" * 65)

    try:
        while True:
            check_and_sync(branch=args.branch)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\n🛑 Auto Git Sync stopped by user.")


if __name__ == "__main__":
    main()
