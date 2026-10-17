"""One-command entry point: `python -m src.run_all`.

Stages are wired in as each task is completed. A stage that is not built yet
is reported as NOT IMPLEMENTED; nothing is faked.
"""

from __future__ import annotations

from src.config import load_config, set_seed

STAGES: list[tuple[str, object]] = [
    # ("1 tidy DCS yields", src.data.dcs.main),          # Task 1
    # ("2 feature extraction", src.data.features.main),  # Task 2
    # ("3 master table", src.data.master.main),          # Task 3
    # ... added task by task
]


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])
    if not STAGES:
        print("No pipeline stages implemented yet (Task 0 scaffold only).")
        return
    for name, fn in STAGES:
        print(f"== {name}")
        fn()


if __name__ == "__main__":
    main()
