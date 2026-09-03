#!/usr/bin/env python3
"""Fetch a CPCB JSON export using an operator-provided endpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.enrichment import fetch_cpcb_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Current CPCB JSON endpoint or exported resource URL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = fetch_cpcb_json(args.url, args.output)
    print({"output": str(args.output), "payload_type": type(payload).__name__})


if __name__ == "__main__":
    main()