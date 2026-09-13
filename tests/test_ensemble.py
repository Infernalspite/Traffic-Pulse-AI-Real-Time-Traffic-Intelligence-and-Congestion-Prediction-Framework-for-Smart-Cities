import torch
from src.models.ensemble import TrafficPulseEnsemble


def test_ensemble_initialization_and_forward():
    ens = TrafficPulseEnsemble(checkpoint_dir="models")
    assert len(ens.loaded_models) >= 5, f"Expected at least 5 models loaded, got {len(ens.loaded_models)}"
    assert "india_aware" in ens.loaded_models
    assert "gwnet" in ens.loaded_models

    # Dummy batch of shape (1, 12, 24, 20)
    x = torch.randn(1, 12, 24, 20)
    res = ens.predict(
        x=x,
        rainfall_mm_hr=20.0,
        waterlogging_depth=0.4,
        festival_intensity=0.8,
        two_wheeler_pct=0.55,
    )

    assert res["ensemble_speeds"].shape == (12, 20)
    assert res["uncertainty_std"].shape == (12, 20)
    assert res["confidence_lower"].shape == (12, 20)
    assert res["confidence_upper"].shape == (12, 20)

    # Weights must sum to approximately 1.0
    total_w = sum(res["weights_used"].values())
    assert abs(total_w - 1.0) < 1e-4

    # Under monsoon and festival, India-Aware model should receive boosted weight
    assert res["weights_used"]["india_aware"] > res["weights_used"]["lstm"]
