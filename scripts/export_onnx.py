#!/usr/bin/env python3
"""Export a trained neural checkpoint for cross-platform inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.traffic_models import AGCRN, DCRNN, GraphWaveNet, STGCN, AdaptiveGraphTemporal, LSTMOnly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = {"lstm": LSTMOnly, "stgcn": STGCN, "dcrnn": DCRNN, "gwnet": GraphWaveNet, "agcrn": AGCRN, "graph": AdaptiveGraphTemporal}
    model = classes[checkpoint["model"]](checkpoint["features"], checkpoint["nodes"], checkpoint["horizon"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    example = torch.zeros(1, 12, checkpoint["features"], checkpoint["nodes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, example, args.output, input_names=["traffic_window"], output_names=["forecast"],
        dynamic_axes={"traffic_window": {0: "batch"}, "forecast": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {args.output}")


if __name__ == "__main__":
    main()