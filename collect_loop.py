import os
import subprocess
import sys
import time
from datetime import datetime, timezone

INTERVAL_SECONDS = int(os.environ.get("TRAFFIC_COLLECTION_INTERVAL_SECONDS", str(15 * 60)))
COLLECTOR = os.path.join(os.path.dirname(__file__), "src", "data", "collector.py")

print("Chennai Traffic Collector running — press Ctrl+C to stop")

while True:
    print(f"[{datetime.now(timezone.utc).isoformat()}] Collecting...")
    result = subprocess.run([sys.executable, COLLECTOR], check=False)
    print(f"Collector exited with status {result.returncode}. Next collection in {INTERVAL_SECONDS} seconds.\n")
    time.sleep(INTERVAL_SECONDS)
