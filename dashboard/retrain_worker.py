"""Background worker: fetch new data → rebuild tensors → retrain models.

Run as a subprocess from the dashboard. Writes structured log lines to stdout
so the dashboard can tail them in real-time.

Usage:
    python dashboard/retrain_worker.py [--epochs N] [--models m1,m2,...] [--skip-collect]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(stage: str, msg: str, pct: int = -1) -> None:
    """Emit a structured JSON log line to stdout."""
    record = {"stage": stage, "msg": msg, "pct": pct, "ts": time.time()}
    print(json.dumps(record), flush=True)


def run(cmd: list[str], stage: str) -> bool:
    """Run a subprocess, streaming its stdout/stderr as log lines."""
    log(stage, f"▶ Running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT),
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(stage, line)
    proc.wait()
    if proc.returncode != 0:
        log(stage, f"❌ Failed with return code {proc.returncode}", pct=-1)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per model (default: 10 for fast retrain)")
    parser.add_argument("--models", type=str, default="gwnet,agcrn,lstm,stgcn,dcrnn,graph,india_aware",
                        help="Comma-separated list of models to retrain")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip live data collection (use existing raw CSV)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    py = sys.executable

    # ── Step 1: Collect fresh traffic data ────────────────────────────────────
    if not args.skip_collect:
        log("collect", "🌐 Fetching latest traffic snapshot from TomTom API...", pct=5)
        ok = run([py, "src/data/collector.py"], stage="collect")
        if not ok:
            log("collect", "⚠️  Data collection failed — retraining on existing data.", pct=10)
    else:
        log("collect", "⏭️  Skipping data collection (--skip-collect flag set).", pct=10)

    # ── Step 2: Enrich with weather + festival data ────────────────────────────
    log("enrich", "🌤️  Running weather & festival enrichment pipeline...", pct=15)
    enrich_script = ROOT / "src" / "data" / "enrichment.py"
    if enrich_script.exists():
        run([py, str(enrich_script)], stage="enrich")
    else:
        log("enrich", "ℹ️  Enrichment script not found — skipping.", pct=20)

    # ── Step 3: Rebuild tensor windows ────────────────────────────────────────
    log("tensors", "🔧 Rebuilding spatio-temporal tensor windows from raw data...", pct=25)
    build_script = ROOT / "scripts" / "build_chennai_tensors.py"
    if build_script.exists():
        ok = run([py, str(build_script)], stage="tensors")
        if not ok:
            log("tensors", "⚠️  Tensor build failed — retraining on previous windows.", pct=30)
    else:
        log("tensors", "ℹ️  Tensor build script not found — using existing windows.", pct=30)

    # ── Step 4: Retrain each model ─────────────────────────────────────────────
    n = len(models)
    for i, model_name in enumerate(models):
        pct_start = 35 + int(i / n * 55)
        pct_end = 35 + int((i + 1) / n * 55)
        log("train", f"🧠 Training [{i+1}/{n}]: {model_name.upper()} ({args.epochs} epochs)...", pct=pct_start)

        windows = ROOT / "data" / "processed" / "chennai_sheet_windows_retrained.npz"
        if not windows.exists():
            windows = ROOT / "data" / "processed" / "chennai_sheet_windows_20260903.npz"

        ckpt = ROOT / "models" / f"retrained_{model_name}_latest.pt"
        out_log = ROOT / "logs" / f"retrained_{model_name}_metrics.json"
        out_log.parent.mkdir(parents=True, exist_ok=True)

        train_script = ROOT / "scripts" / "train_traffic_models.py"
        if train_script.exists() and windows.exists():
            ok = run(
                [py, str(train_script),
                 "--windows", str(windows),
                 "--model", model_name,
                 "--epochs", str(args.epochs),
                 "--batch-size", "64",
                 "--checkpoint", str(ckpt),
                 "--output", str(out_log)],
                stage="train",
            )
            if ok:
                log("train", f"✅ {model_name.upper()} checkpoint saved → {ckpt.name}", pct=pct_end)
            else:
                log("train", f"⚠️  {model_name.upper()} training failed — previous checkpoint kept.", pct=pct_end)
        else:
            log("train", f"⚠️  Training script or windows not found for {model_name} — skipping.", pct=pct_end)

    # ── Step 5: Done ───────────────────────────────────────────────────────────
    log("done", "🎉 Retrain complete! Reload the dashboard to use the updated models.", pct=100)


if __name__ == "__main__":
    main()
