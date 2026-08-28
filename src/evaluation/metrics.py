"""Shared forecasting metrics and India-stratified evaluation."""
from __future__ import annotations
import numpy as np

def _safe_mape(actual,predicted):
    actual=np.asarray(actual,dtype=float); predicted=np.asarray(predicted,dtype=float); mask=np.abs(actual)>1e-8
    return float(np.mean(np.abs((actual[mask]-predicted[mask])/actual[mask]))*100) if mask.any() else float("nan")

def regression_metrics(actual,predicted):
    actual=np.asarray(actual,dtype=float); predicted=np.asarray(predicted,dtype=float); error=actual-predicted
    return {"MAE":float(np.mean(np.abs(error))),"RMSE":float(np.sqrt(np.mean(error**2))),"MAPE":_safe_mape(actual,predicted)}

def horizon_metrics(actual,predicted,steps=(3,6,12),step_minutes=5):
    actual=np.asarray(actual); predicted=np.asarray(predicted); output={}
    if actual.shape!=predicted.shape: raise ValueError(f"shape mismatch: {actual.shape} != {predicted.shape}")
    for step in steps:
        if step>actual.shape[1]: continue
        output[f"{step*step_minutes}min"]=regression_metrics(actual[:,step-1:step],predicted[:,step-1:step])
    return output

def stratified_metrics(actual,predicted,strata:dict[str,np.ndarray],steps=(3,6,12),step_minutes=5):
    actual=np.asarray(actual); predicted=np.asarray(predicted); results={}
    for name,mask in strata.items():
        mask=np.asarray(mask,dtype=bool)
        if mask.shape[0]!=actual.shape[0]: raise ValueError(f"stratum {name} has wrong sample count")
        results[name]=horizon_metrics(actual[mask],predicted[mask],steps,step_minutes) if mask.any() else None
    return results

def compare_models(predictions:dict[str,np.ndarray],actual:np.ndarray,steps=(3,6,12),step_minutes=5):
    return {name:horizon_metrics(actual,prediction,steps,step_minutes) for name,prediction in predictions.items()}
