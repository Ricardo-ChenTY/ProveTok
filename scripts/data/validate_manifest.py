from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as `python scripts/data/*.py ...`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provetok.data.manifest_schema import load_manifest, summarize_manifest
from provetok.data.protocol_lock import ProtocolLock


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a ProveTok manifest (.jsonl/.csv)")
    ap.add_argument("--manifest", type=str, required=True, help="Path to manifest.jsonl or manifest.csv")
    ap.add_argument("--no-protocol-lock", action="store_true", help="Skip ProtocolLock checks")
    args = ap.parse_args()

    records = load_manifest(args.manifest)
    summary = summarize_manifest(records)
    print(json.dumps(summary, indent=2))

    if not args.no_protocol_lock:
        ProtocolLock().validate_or_die(records)
        print("\n[OK] ProtocolLock passed")


if __name__ == "__main__":
    main()
