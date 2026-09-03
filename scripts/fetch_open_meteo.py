#!/usr/bin/env python3
"""Fetch hourly Open-Meteo historical rainfall and visibility."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.enrichment import fetch_open_meteo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = fetch_open_meteo(args.latitude, args.longitude, args.start_date, args.end_date, args.output)
    print({"rows": len(frame), "output": str(args.output)})


if __name__ == "__main__":
    main()