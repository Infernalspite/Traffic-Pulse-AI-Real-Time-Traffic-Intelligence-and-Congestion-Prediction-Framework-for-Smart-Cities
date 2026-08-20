import subprocess
import time
from datetime import datetime

interval = 15 * 60  # 15 minutes in seconds

print("Chennai Traffic Collector running — press Ctrl+C to stop")
print("Run this only when plugged in!\n")

while True:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Collecting...")
    subprocess.run(['python', 'src/data/collector.py'])
    print(f"Done. Next collection in 15 minutes.\n")
    time.sleep(interval)