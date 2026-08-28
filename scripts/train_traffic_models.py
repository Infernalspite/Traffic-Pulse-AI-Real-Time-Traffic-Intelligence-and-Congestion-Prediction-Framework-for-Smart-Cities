#!/usr/bin/env python3
"""Train one traffic model with the shared windows and evaluation contract."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from src.evaluation.metrics import horizon_metrics
from src.models.traffic_models import AdaptiveGraphTemporal,LSTMOnly,Persistence

def run_epoch(model,loader,optimizer,device,training):
    loss_fn=torch.nn.L1Loss(); model.train(training); total=0.0
    for features,target in loader:
        features,target=features.to(device),target.to(device); prediction=model(features); loss=loss_fn(prediction,target)
        if training: optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
        total+=float(loss.detach())*len(features)
    return total/len(loader.dataset)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--windows",type=Path,default=Path("data/processed/chennai_windows.npz")); parser.add_argument("--model",choices=["persistence","lstm","graph"],default="graph"); parser.add_argument("--epochs",type=int,default=100); parser.add_argument("--batch-size",type=int,default=32); parser.add_argument("--output",type=Path,default=Path("logs/traffic_model_metrics.json")); args=parser.parse_args()
    data=np.load(args.windows,allow_pickle=True); x_train=torch.tensor(data["X_train"],dtype=torch.float32); x_val=torch.tensor(data["X_val"],dtype=torch.float32); x_test=torch.tensor(data["X_test"],dtype=torch.float32); y_train=torch.tensor(data["y_train"],dtype=torch.float32); y_val=torch.tensor(data["y_val"],dtype=torch.float32); y_test=torch.tensor(data["y_test"],dtype=torch.float32)
    nodes=x_train.shape[-1]; features=x_train.shape[2]; horizon=y_train.shape[1]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model=="persistence":
        prediction=Persistence(horizon).predict(x_test).numpy()
    else:
        model=(LSTMOnly(features,nodes,horizon) if args.model=="lstm" else AdaptiveGraphTemporal(features,nodes,horizon)).to(device); train_loader=DataLoader(TensorDataset(x_train,y_train),batch_size=args.batch_size,shuffle=False); val_loader=DataLoader(TensorDataset(x_val,y_val),batch_size=args.batch_size); optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=50); best=float("inf"); best_state=None; patience=20; stale=0
        for epoch in range(args.epochs):
            run_epoch(model,train_loader,optimizer,device,True); scheduler.step(epoch); validation=run_epoch(model,val_loader,optimizer,device,False)
            if validation<best: best=validation; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
            else: stale+=1
            if stale>=patience: break
        model.load_state_dict(best_state); model.eval(); with_context=torch.no_grad()
        with torch.no_grad(): prediction=model(x_test.to(device)).cpu().numpy()
    mean=float(data["feature_mean"][0]); scale=float(data["feature_scale"][0]); actual=y_test.numpy()*scale+mean; prediction=prediction*scale+mean
    result={"model":args.model,"device":str(device),"test_samples":len(actual),"metrics":horizon_metrics(actual,prediction)}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
