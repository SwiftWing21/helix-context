"""Initialize main.db — the routing layer for sharded genomes.

Idempotent. Safe to run on an existing main.db; schema is additive.

Usage:
    python scripts/init_main_db.py [--path F:/Projects/helix-context/genomes/main.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from helix_context.shard_schema import init_main_db, open_main_db, list_shards


DEFAULT_PATH = "F:/Projects/helix-context/genomes/main.db"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    args = ap.parse_args()

    print(f"[init-main] target: {args.path}")
    conn = open_main_db(args.path)
    init_main_db(conn)

    shards = list_shards(conn)
    print(f"[init-main] ok. shards registered: {len(shards)}")
    for row in shards:
        print(f"  {row['category']:>12}  {row['shard_name']}  ({row['path']})")

    conn.close()
    print(f"[init-main] {args.path} ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
