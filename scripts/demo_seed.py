"""Load a believable week of made-up memories into the separate demo database (demo/data/kin.db).

The real database (data/kin.db) is never opened. Run Kin on the demo data with `kin run --demo`; the
UI's "Reset demo" button wipes the demo database and loads the week again.

  uv run python scripts/demo_seed.py              # create or reset the demo database
  uv run python scripts/demo_seed.py --profile pi5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kin.config import PROFILES, load_config, project_root  # noqa: E402
from kin.memory.demo_week import seed_demo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=PROFILES, default="laptop")
    args = ap.parse_args()
    cfg = load_config(args.profile)
    cfg.data_dir, cfg.demo = project_root() / "demo", True
    print(seed_demo(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
