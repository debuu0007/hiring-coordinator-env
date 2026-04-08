from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    root_server_path = Path(__file__).resolve().parents[1] / "server.py"
    spec = importlib.util.spec_from_file_location("hiring_env_root_server", root_server_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load root server module from {root_server_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
