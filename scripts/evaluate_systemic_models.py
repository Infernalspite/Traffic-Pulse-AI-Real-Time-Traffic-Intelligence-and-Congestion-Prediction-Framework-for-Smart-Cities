#!/usr/bin/env python3
"""Evaluate shared-feature systemic-risk tabular baselines."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def metrics(y, pred, score):
    return {"Accuracy": round(float(accuracy_score(y, pred)), 4), "Precision": round(float(precision_score(y, pred, zero_division=0)), 4), "Recall": round(float(recall_score(y, pred, zero_division=0)), 4), "F1": round(float(f1_score(y, pred, zero_division=0)), 4), "ROC-AUC": round(float(roc_auc_score(y, score)), 4) if len(set(y)) > 1 else None}

def evaluate(frame, date_column, target, train_end):
    frame=frame.copy(); frame[date_column]=pd.to_datetime(frame[date_column], errors="coerce", utc=True); frame=frame.dropna(subset=[date_column,target]).sort_values(date_column)
    train=frame[frame[date_column] < train_end]; test=frame[frame[date_column] >= train_end]
    if train.empty or test.empty: raise ValueError("Time split produced an empty train or test set")
    X_train=train.drop(columns=[date_column,target]); X_test=test.drop(columns=[date_column,target]); y_train=train[target].astype(int); y_test=test[target].astype(int)
    numeric=list(X_train.select_dtypes(include="number").columns); X_train=X_train[numeric]; X_test=X_test[numeric]
    models={"Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)), "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"]=make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42))
    except ImportError: pass
    results=[]
    for name, model in models.items():
        model.fit(X_train,y_train); pred=model.predict(X_test); score=model.predict_proba(X_test)[:,1]; results.append({"Model":name, **metrics(y_test,pred,score)})
    return {"target":target,"date_column":date_column,"train_end":train_end.isoformat(),"train_rows":len(train),"test_rows":len(test),"positive_train":int(y_train.sum()),"positive_test":int(y_test.sum()),"results":results}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--features",type=Path,default=Path("data/systemic_risk/features.csv")); parser.add_argument("--target",default="high_stress_next_30d"); parser.add_argument("--date-column",default="date"); parser.add_argument("--train-end",required=True); parser.add_argument("--output",type=Path,default=Path("logs/systemic_risk_model_comparison.json")); args=parser.parse_args()
    result=evaluate(pd.read_csv(args.features),args.date_column,args.target,pd.Timestamp(args.train_end,tz="UTC")); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
