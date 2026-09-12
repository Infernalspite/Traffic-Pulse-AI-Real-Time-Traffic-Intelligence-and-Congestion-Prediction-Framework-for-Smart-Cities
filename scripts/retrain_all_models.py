#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

MODELS = ['stgcn', 'dcrnn', 'gwnet', 'agcrn', 'graph']
WINDOWS = Path('data/processed/chennai_sheet_windows_retrained.npz')

sep = '=' * 50
for model_name in MODELS:
    ckpt = Path('models') / f'retrained_{model_name}_latest.pt'
    out = Path('logs') / f'retrained_{model_name}_metrics.json'
    print(f'\n{sep}\nTraining {model_name} on GPU...\n{sep}')
    cmd = [
        sys.executable,
        'scripts/train_traffic_models.py',
        '--windows', str(WINDOWS),
        '--model', model_name,
        '--epochs', '25',
        '--batch-size', '64',
        '--checkpoint', str(ckpt),
        '--output', str(out)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f'Error training {model_name}: {res.stderr}')
