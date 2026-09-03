#!/usr/bin/env python3
"""Download benchmark files when the operator supplies the current URLs."""
from __future__ import annotations

import argparse
from pathlib import Path
import requests


def download(url: str, output: Path) -> None:
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metr-la-url")
    parser.add_argument("--pems-bay-url")
    parser.add_argument("--output-dir", type=Path, default=Path("data/international"))
    args = parser.parse_args()
    if not args.metr_la_url and not args.pems_bay_url:
        parser.error("provide --metr-la-url and/or --pems-bay-url")
    if args.metr_la_url:
        download(args.metr_la_url, args.output_dir / "metr-la.download")
    if args.pems_bay_url:
        download(args.pems_bay_url, args.output_dir / "pems-bay.download")
    print(f"benchmarks saved under {args.output_dir}")


if __name__ == "__main__":
    main()