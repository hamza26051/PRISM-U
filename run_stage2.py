from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="PRISM-U Phase 2 end-to-end runner")
    p.add_argument("--config", default="configs/stage2.yaml")
    p.add_argument("--phase", choices=["all", "bank", "responses", "diagnostics", "validate"], default="all")
    p.add_argument("--role", choices=["all", "surrogate", "validation", "final"], default="all")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def run(cmd):
    print("\n$ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    root = Path.cwd().resolve()
    required = [root / "scripts", root / "prismu_stage2", root / "configs" / "stage2.yaml"]
    if not all(p.exists() for p in required):
        raise RuntimeError("Run this command from the extracted PRISM-U Phase 2 code package root.")
    py = sys.executable

    if args.phase in {"all", "bank"}:
        cmd = [py, "scripts/build_bank.py", "--config", args.config, "--role", args.role]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.device:
            cmd += ["--device", args.device]
        if args.force:
            cmd += ["--force"]
        run(cmd)
        if args.phase == "all" and (args.role != "all" or args.seed is not None):
            print("Targeted bank run complete. Re-run Phase 2 after every requested family exists.")
            return

    if args.phase in {"all", "responses"}:
        cmd = [py, "scripts/cache_responses.py", "--config", args.config, "--role", args.role]
        if args.device:
            cmd += ["--device", args.device]
        if args.force:
            cmd += ["--force"]
        run(cmd)

    if args.phase in {"all", "diagnostics"}:
        run([py, "scripts/diagnostics.py", "--config", args.config])

    if args.phase in {"all", "validate"}:
        run([py, "scripts/validate_stage2.py", "--config", args.config])


if __name__ == "__main__":
    main()
