import subprocess
import cv2
import numpy as np

width, height = 1920, 1080
command = [
    'ffmpeg',
    '-rtsp_transport', 'tcp',
    '-i', 'rtsp://100.108.97.81:8554/cam1',
    '-frames:v', '1',
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo',
    '-loglevel', 'quiet',
    '-'
]

pipe = subprocess.Popen(command, stdout=subprocess.PIPE)
raw_image = pipe.stdout.read(width * height * 3)
frame = np.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3))
cv2.imwrite("frame.jpg", frame)
print("✅ Frame saved as frame.jpg")
pipe.terminate()

