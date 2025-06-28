import sys

#Importing path to generic classes
sys.path.append( '../' )

from System.App.CameraMonitor.CameraMonitor import CameraMonitor
import time

monitor = CameraMonitor(port=5151)
monitor.start()
print(f"Camera Monitor is running on {monitor.get_url()}")
# Simulate camera state changes
states = ['active', 'inactive', 'recording']
import random

while True:
    time.sleep(0.01)
    pass