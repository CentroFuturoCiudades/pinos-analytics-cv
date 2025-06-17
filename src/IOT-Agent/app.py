import sys
sys.path.append('../')
from System.App.CameraMonitor.CameraMonitor import CameraMonitor
import time

monitor = CameraMonitor(port=5151, api_port=5152)
monitor.start()
print(f"Camera Monitor is running on {monitor.get_url()}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down.")