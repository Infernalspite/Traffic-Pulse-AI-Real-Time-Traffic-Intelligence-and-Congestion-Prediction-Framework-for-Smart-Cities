"""CLI Tool to generate multi-model ensemble traffic predictions accounting for all parameters.

Usage:
    python scripts/run_ensemble_prediction.py --rain 25 --waterlog 0.4 --festival 0.8 --horizon 15
"""
import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from src.models.ensemble import TrafficPulseEnsemble

# Ensure safe UTF-8 printing on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FREE_FLOW_DEFAULTS = {
    "Kathipara": 35.0, "Guindy": 35.0, "T_Nagar_Panagal": 30.0,
    "Anna_Salai_Teynampet": 35.0, "Central_Station": 28.0, "Koyambedu": 25.0,
    "Ashok_Nagar": 32.0, "Adyar_Signal": 30.0, "Velachery_Junction": 30.0,
    "Tambaram": 35.0, "Perungudi_OMR": 40.0, "Thoraipakkam_OMR": 40.0,
    "Porur": 30.0, "Anna_Nagar_Roundtana": 28.0, "Perambur": 30.0,
    "Mylapore": 25.0, "Chromepet": 35.0, "Kelambakkam_OMR": 45.0,
    "Egmore": 30.0, "Vadapalani": 28.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ensemble traffic prediction accounting for all parameters.")
    parser.add_argument("--rain", type=float, default=0.0, help="Monsoon rain intensity (mm/hr)")
    parser.add_argument("--waterlog", type=float, default=0.0, help="Waterlogging severity (0.0 to 1.0)")
    parser.add_argument("--festival", type=float, default=0.0, help="Festival calendar intensity (0.0 to 1.0)")
    parser.add_argument("--two-wheeler", type=float, default=0.45, help="2-Wheeler modal share (0.0 to 1.0)")
    parser.add_argument("--horizon", type=int, default=15, choices=[15, 30, 45, 60], help="Forecast horizon minutes")
    parser.add_argument("--closure", type=str, default="None", help="Junction name to simulate complete closure")
    parser.add_argument("--output-json", type=str, default=None, help="Save predictions to JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("🚦 TRAFFIC PULSE AI — MULTI-MODEL ENSEMBLE FORECASTING ENGINE")
    print("=" * 80)
    print(f"Parameters Accounted For:")
    print(f"  • Monsoon Rainfall:        {args.rain:.1f} mm/hr")
    print(f"  • Waterlogging Depth:      {args.waterlog:.2f} (scale 0-1)")
    print(f"  • Festival Calendar Rush:  {args.festival:.2f} (scale 0-1)")
    print(f"  • 2-Wheeler Modal Share:   {args.two_wheeler * 100:.1f}%")
    print(f"  • Forecast Horizon:        +{args.horizon} minutes")
    print(f"  • Simulated Closure:       {args.closure}")
    print("-" * 80)

    # Initialize ensemble
    ensemble = TrafficPulseEnsemble(checkpoint_dir=ROOT_DIR / "models")
    print(f"Models Combined ({len(ensemble.loaded_models)} active): {', '.join(ensemble.loaded_models)}")

    # Load test window
    windows_path = ROOT_DIR / "data" / "processed" / "chennai_sheet_windows_retrained.npz"
    if not windows_path.exists():
        windows_path = ROOT_DIR / "data" / "processed" / "chennai_sheet_windows_20260903.npz"

    if windows_path.exists():
        npz = np.load(windows_path, allow_pickle=True)
        junctions = list(npz["junctions"])
        scale = float(npz["feature_scale"][0])
        mean = float(npz["feature_mean"][0])
        x = torch.tensor(npz["X_test"][0:1], dtype=torch.float32)
    else:
        junctions = list(FREE_FLOW_DEFAULTS.keys())
        scale, mean = 7.5, 26.0
        x = torch.randn(1, 12, 24, len(junctions))

    # Inject environmental and contextual parameters into input tensor
    x[:, :, 12, :] = args.rain
    x[:, :, 14, :] = args.waterlog
    x[:, :, 15, :] = 1.0 if args.festival > 0 else 0.0
    x[:, :, 16, :] = args.festival
    x[:, :, 4, :] = args.two_wheeler

    # Run inference across all models
    res = ensemble.predict(
        x=x,
        rainfall_mm_hr=args.rain,
        waterlogging_depth=args.waterlog,
        festival_intensity=args.festival,
        two_wheeler_pct=args.two_wheeler,
        scale=scale,
        mean=mean,
    )

    # Step index for requested horizon (each step = 5 min)
    step_idx = min(11, max(0, (args.horizon // 5) - 1))

    # Contextual penalty multipliers
    monsoon_decay = max(0.4, 1.0 - 0.35 * (1.0 - np.exp(-args.rain / 15.0)))
    wl_penalty = max(0.45, 1.0 - 0.5 * args.waterlog)
    fest_penalty = max(0.65, 1.0 - 0.3 * args.festival)
    mix_penalty = 1.0 - max(0.0, (args.two_wheeler - 0.45) * 0.003)
    total_penalty = monsoon_decay * wl_penalty * fest_penalty * mix_penalty

    ensemble_speeds = res["ensemble_speeds"][step_idx] * total_penalty
    uncertainty_std = res["uncertainty_std"][step_idx]

    print("\nModel Weights in Ensemble Consensus:")
    for m_name, w in sorted(res["weights_used"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {m_name:<16}: {w * 100:.1f}%")

    print("\n" + "=" * 96)
    print(f"{'Junction':<22} | {'FreeFlow':<8} | {'Ensemble (km/h)':<16} | {'95% Conf Interval':<19} | {'Capacity':<9} | {'Status'}")
    print("=" * 96)

    output_rows = []
    alerts_count = 0

    for i, j_name in enumerate(junctions):
        ff = FREE_FLOW_DEFAULTS.get(j_name, 35.0)
        spd = float(ensemble_speeds[i])
        std = float(uncertainty_std[i])

        if args.closure == j_name:
            spd = 3.5
            std = 0.5

        cap = min(100.0, max(5.0, (1.0 - spd / ff) * 100.0))
        c_low = max(2.0, spd - 1.96 * std)
        c_high = spd + 1.96 * std
        is_alert = cap >= 80.0

        if is_alert:
            status = "🚨 SEVERE GRIDLOCK (>80%)"
            alerts_count += 1
        elif cap >= 50.0:
            status = "⚠️ MODERATE SLOWDOWN"
        else:
            status = "🟢 FREE FLOW"

        print(f"{j_name:<22} | {ff:<8.1f} | {spd:<16.2f} | [{c_low:<5.1f}, {c_high:<5.1f}] km/h | {cap:<8.1f}% | {status}")

        output_rows.append({
            "junction": j_name,
            "free_flow_speed": ff,
            "ensemble_speed_kmh": round(spd, 2),
            "uncertainty_std_kmh": round(std, 2),
            "confidence_95_interval": [round(c_low, 2), round(c_high, 2)],
            "capacity_saturation_pct": round(cap, 1),
            "alert_triggered": is_alert,
            "individual_models_kmh": {
                m: round(float(res["model_predictions"][m][step_idx, i] * total_penalty), 2)
                for m in ensemble.loaded_models
            }
        })

    print("=" * 96)
    print(f"Summary: {alerts_count} junctions triggered >80% capacity alert under these conditions.")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "parameters": vars(args),
            "ensemble_weights": res["weights_used"],
            "predictions": output_rows,
        }
        out_p.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"Saved full prediction JSON to: {out_p}")


if __name__ == "__main__":
    main()
