"""Deterministic Chennai traffic tensor preparation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

FEATURE_NAMES=["speed","volume","occupancy","density","two_wheeler_pct","auto_pct","bus_pct","car_pct","hour_sin","hour_cos","dow_sin","dow_cos","rainfall_mm_hr","visibility","waterlogging","festival_indicator","festival_intensity","days_relative_festival","pm25","pm10","road_width_category","signal_type","flyover_presence","incident_flag"]

@dataclass(frozen=True)
class TensorConfig:
    input_steps:int=12
    forecast_steps:int=12
    train_fraction:float=0.70
    validation_fraction:float=0.10
    interpolation_minutes:float=10.0

def _normalise_columns(frame:pd.DataFrame)->pd.DataFrame:
    frame=frame.copy(); frame.columns=[str(c).strip().lower().replace(" ","_") for c in frame.columns]
    aliases={"time":"timestamp","junction_name":"junction","total_vehicle":"total_vehicles","precipitation":"precipitation_mm"}
    return frame.rename(columns={k:v for k,v in aliases.items() if k in frame})

def load_traffic(path:Path)->pd.DataFrame:
    frame=pd.read_parquet(path) if path.suffix.lower() in {".parquet",".pq"} else pd.read_csv(path); frame=_normalise_columns(frame)
    for column in ["timestamp","junction"]:
        if column not in frame: raise ValueError(f"traffic data missing {column}")
    frame["timestamp"]=pd.to_datetime(frame["timestamp"],errors="coerce",utc=True); frame["junction"]=frame["junction"].astype(str).str.strip()
    return frame.dropna(subset=["timestamp","junction"]).drop_duplicates(["timestamp","junction"],keep="last").sort_values(["timestamp","junction"])

def _num(frame,column,default=np.nan):
    return pd.to_numeric(frame[column],errors="coerce") if column in frame else pd.Series(default,index=frame.index,dtype=float)

def _pct(frame,column,total):
    values=_num(frame,column); return values.div(total.replace(0,np.nan)).mul(100)

def to_feature_frame(frame:pd.DataFrame)->tuple[pd.DataFrame,list[str]]:
    frame=_normalise_columns(frame.copy()); total=_num(frame,"total_vehicles")
    if total.isna().all(): total=sum((_num(frame,c).fillna(0) for c in ["two_wheeler","autorickshaw","car","bus","lcv","truck","bicycle"]),pd.Series(0,index=frame.index,dtype=float))
    time=pd.to_datetime(frame["timestamp"],utc=True)
    out=pd.DataFrame({"timestamp":time,"junction":frame["junction"].astype(str),"speed":_num(frame,"current_speed"),"volume":total,"occupancy":_num(frame,"occupancy"),"density":_num(frame,"pcu_density"),"two_wheeler_pct":_pct(frame,"two_wheeler",total),"auto_pct":_pct(frame,"autorickshaw",total),"bus_pct":_pct(frame,"bus",total),"car_pct":_pct(frame,"car",total),"hour_sin":np.sin(2*np.pi*time.dt.hour/24),"hour_cos":np.cos(2*np.pi*time.dt.hour/24),"dow_sin":np.sin(2*np.pi*time.dt.dayofweek/7),"dow_cos":np.cos(2*np.pi*time.dt.dayofweek/7),"rainfall_mm_hr":_num(frame,"rain_mm",_num(frame,"precipitation_mm")),"visibility":_num(frame,"visibility"),"waterlogging":_num(frame,"waterlogging",_num(frame,"road_closure",0)).fillna(0).astype(float),"festival_indicator":_num(frame,"is_festival",0).fillna(0),"festival_intensity":_num(frame,"festival_impact",0).fillna(0),"days_relative_festival":_num(frame,"days_relative_festival"),"pm25":_num(frame,"pm25",_num(frame,"pm2_5")),"pm10":_num(frame,"pm10"),"road_width_category":_num(frame,"road_width_category",0).fillna(0),"signal_type":_num(frame,"signal_type",0).fillna(0),"flyover_presence":_num(frame,"flyover_presence",0).fillna(0),"incident_flag":_num(frame,"incident_flag",_num(frame,"road_closure",0)).fillna(0)})
    return out.sort_values(["timestamp","junction"]),FEATURE_NAMES

def impute_features(frame:pd.DataFrame,adjacency:np.ndarray|None,config:TensorConfig)->pd.DataFrame:
    result=frame.copy().set_index(["timestamp","junction"]); junctions=list(result.index.get_level_values(1).unique()); numeric=[c for c in FEATURE_NAMES if c in result]
    for junction in junctions:
        key=(slice(None),junction); part=result.loc[key,numeric].copy(); part.index=pd.DatetimeIndex(part.index.get_level_values(0)); part=part[~part.index.duplicated(keep="last")].sort_index(); part=part.interpolate(method="time",limit_area="inside",limit_direction="both")
        result.loc[key,numeric]=part.to_numpy()
    if adjacency is not None:
        adjacency=np.asarray(adjacency,dtype=float); np.fill_diagonal(adjacency,0); adjacency=adjacency/(adjacency.sum(axis=1,keepdims=True)+1e-12)
        for timestamp in result.index.get_level_values(0).unique():
            rows=result.loc[timestamp,numeric]; values=rows.to_numpy(dtype=float)
            for col in range(values.shape[1]):
                missing=np.isnan(values[:,col]); values[missing,col]=(adjacency@np.nan_to_num(values[:,col],nan=0))[missing]
            result.loc[timestamp,numeric]=values
    return result.reset_index()

def _split_dates(dates:pd.DatetimeIndex,config:TensorConfig)->tuple[pd.Timestamp,pd.Timestamp]:
    unique=pd.DatetimeIndex(sorted(pd.unique(dates))); train_end=unique[int(len(unique)*config.train_fraction)]; validation_end=unique[int(len(unique)*(config.train_fraction+config.validation_fraction))]; return train_end,validation_end

def make_windows(frame:pd.DataFrame,config:TensorConfig,adjacency:np.ndarray|None=None)->dict:
    frame=impute_features(frame,adjacency,config); frame=frame.sort_values(["timestamp","junction"]); junctions=sorted(frame["junction"].unique()); timestamps=sorted(frame["timestamp"].unique()); train_end,validation_end=_split_dates(pd.DatetimeIndex(timestamps),config)
    by_time=frame.pivot(index="timestamp",columns="junction",values=FEATURE_NAMES).reindex(columns=pd.MultiIndex.from_product([FEATURE_NAMES,junctions])).sort_index()
    values=by_time.to_numpy(dtype=float).reshape(len(by_time),len(FEATURE_NAMES),len(junctions)); values=np.nan_to_num(values,nan=0.0); xs=[]; ys=[]; dates=[]
    horizon=config.input_steps+config.forecast_steps
    for start in range(0,len(values)-horizon+1): xs.append(values[start:start+config.input_steps]); ys.append(values[start+config.input_steps:start+horizon,0,:,:]); dates.append(by_time.index[start+config.input_steps+config.forecast_steps-1])
    x=np.asarray(xs,dtype=np.float32); y=np.asarray(ys,dtype=np.float32); dates=pd.DatetimeIndex(dates); train=x[dates<train_end]; val=x[(dates>=train_end)&(dates<validation_end)]; test=x[dates>=validation_end]
    return {"X_train":train,"X_val":val,"X_test":test,"y_train":y[dates<train_end],"y_val":y[(dates>=train_end)&(dates<validation_end)],"y_test":y[dates>=validation_end],"junctions":np.asarray(junctions),"feature_names":np.asarray(FEATURE_NAMES),"train_end":str(train_end),"validation_end":str(validation_end)}

def save_windows(windows:dict,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**windows); path.with_suffix(".json").write_text(json.dumps({k:(v.tolist() if isinstance(v,np.ndarray) and v.ndim==1 else v) for k,v in windows.items() if isinstance(v,(str,int,float,list)) or (isinstance(v,np.ndarray) and v.ndim==1)},indent=2))
