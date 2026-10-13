from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="PRISM-U Stage 2 end-to-end runner")
    p.add_argument("--config", default="stage2/configs/stage2.yaml")
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
    repo_root = Path.cwd().resolve()
    if not (repo_root / "stage2").is_dir():
        raise RuntimeError("Run this command from the PRISM-U repository root.")
    py = sys.executable

    if args.phase in {"all", "bank"}:
        cmd = [py, "stage2/scripts/build_bank.py", "--config", args.config, "--role", args.role]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.device:
            cmd += ["--device", args.device]
        if args.force:
            cmd += ["--force"]
        run(cmd)
        # A targeted bank run is intentionally allowed to stop here. It is the normal
        # Colab workflow when one seed/family is trained per session.
        if args.phase == "all" and (args.role != "all" or args.seed is not None):
            print("Targeted bank run complete. Re-run --phase all with no --role/--seed after every family exists.")
            return

    if args.phase in {"all", "responses"}:
        cmd = [py, "stage2/scripts/cache_responses.py", "--config", args.config, "--role", args.role]
        if args.device:
            cmd += ["--device", args.device]
        if args.force:
            cmd += ["--force"]
        run(cmd)

    if args.phase in {"all", "diagnostics"}:
        run([py, "stage2/scripts/diagnostics.py", "--config", args.config])

    if args.phase in {"all", "validate"}:
        run([py, "stage2/scripts/validate_stage2.py", "--config", args.config])


if __name__ == "__main__":
    main()
