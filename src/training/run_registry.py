"""File-backed experiment registry that works offline and in CI."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def record_run(path: Path, config: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")