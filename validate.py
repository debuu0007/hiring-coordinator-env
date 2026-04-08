from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str], required: bool = True) -> int:
    print(f"$ {' '.join(command)}")
    result = subprocess.run(command, cwd=ROOT)
    if required and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def main() -> None:
    run([sys.executable, "-m", "pytest", "tests"])

    if shutil.which("openenv"):
        run(["openenv", "validate"])
    else:
        print("openenv CLI not found; install openenv-core and run `openenv validate` before submission.")


if __name__ == "__main__":
    main()
