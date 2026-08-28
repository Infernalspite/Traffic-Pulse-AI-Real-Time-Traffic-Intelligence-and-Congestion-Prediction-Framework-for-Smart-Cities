#!/usr/bin/env python3
"""Create train/validation/test traffic tensors from a CSV."""
import argparse
from pathlib import Path
import numpy as np
from src.data.pipeline import TensorConfig, load_traffic, to_feature_frame, make_windows, save_windows

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--input",type=Path,default=Path("data/processed/chennai_full_dataset.csv")); parser.add_argument("--output",type=Path,default=Path("data/processed/chennai_windows.npz")); parser.add_argument("--adjacency",type=Path); args=parser.parse_args()
    frame=load_traffic(args.input); features,_=to_feature_frame(frame); adjacency=np.load(args.adjacency) if args.adjacency else None; windows=make_windows(features,TensorConfig(),adjacency); save_windows(windows,args.output); print({key:(value.shape if hasattr(value,"shape") else value) for key,value in windows.items()})

if __name__=="__main__": main()
