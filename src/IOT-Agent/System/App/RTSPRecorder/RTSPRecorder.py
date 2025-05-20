from threading import Thread
import cv2
import os
import datetime
import ffmpegcv
import time
import imutils
import argparse
import numpy as np
from Generic.Global.Borg import Borg

class RTSPRecorder(Borg):
    __ctx = None
    __config = None

    def __init__(self, camera='camera1', folder='', codec='libx264', width=None, height=None, fps=None, verbose=False, visualize=False):
        self.ctx = Borg._Borg__shared_state['ctx']
        src = self.ctx['__obj']['__config'].get('rtsp')[camera]
        self.ctx['__obj']['__log'].setLog(f"[INFO] Reading from {src}")

        self.ctx_rtsp = {
            "camera": camera,
            "capture": cv2.VideoCapture(src, cv2.CAP_FFMPEG),
            "output_video": None,
            "folder": folder,
            "fps": fps,
            "width": width,
            "height": height,
            "codec": codec,
            "verbose": verbose
        }

        self.visual = visualize
        self.queued_seconds = 3
        self.minimum_frames = 30
        self.video_queue = []
        self.record_queue = []
        self.writing_video = False
        self.active = True
        self.recording = False
        self.record_ongoing = False
        self.allow_stop = False

        cap = self.ctx_rtsp['capture']
        if self.ctx_rtsp['fps'] is None:
            self.ctx_rtsp['fps'] = cap.get(cv2.CAP_PROP_FPS)
        if self.ctx_rtsp['width'] is None:
            self.ctx_rtsp['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if self.ctx_rtsp['height'] is None:
            self.ctx_rtsp['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.mutithreadingRead()

    def mutithreadingRead(self):
        self.ctx['__obj']['__log'].setLog('Starting multithreading')
        self.readStatus, frame = self.ctx_rtsp["capture"].read()
        if not self.readStatus:
            self.ctx['__obj']['__log'].setLog("[ERROR] Failed to read from video stream")
            return
        self.frame = self.resize(frame)
        self.cameraThread = Thread(target=self.update_queue)
        self.cameraThread.daemon = True
        self.readStatus = False
        self.cameraThread.start()
        if self.visual:
            self.show_video()

    def resize(self, frame):
        if self.ctx_rtsp["width"] or self.ctx_rtsp["height"]:
            return imutils.resize(frame, width=self.ctx_rtsp["width"], height=self.ctx_rtsp["height"])
        return frame

    def update_queue(self):
        timemark = time.perf_counter()
        self.ctx['__obj']['__log'].setLog(f"[INFO] Starting the capturing process in the {self.ctx_rtsp['camera']}")
        time_inactivity = 0
        recorded_frames = 0

        while self.active:
            if self.ctx_rtsp["capture"].isOpened():
                try:
                    self.readStatus, frame = self.ctx_rtsp["capture"].read()
                    if not self.readStatus or frame is None:
                        continue
                    self.frame = self.resize(frame)
                    self.video_queue.append(self.frame)
                    if len(self.video_queue) > self.queued_seconds * self.ctx_rtsp['fps']:
                        self.video_queue.pop(0)

                    if self.recording:
                        if not self.record_queue:
                            self.record_queue = self.video_queue.copy()
                        self.record_queue.append(self.frame)

                    if self.record_ongoing:
                        if self.record_queue:
                            self.writing_video = True
                            record_frame = self.record_queue.pop(0)
                            self.ctx_rtsp['output_video'].write(record_frame)
                            recorded_frames += 1
                            self.writing_video = False
                        elif self.allow_stop:
                            self.record_ongoing = False
                            self.ctx_rtsp['output_video'].release()
                            if recorded_frames < self.minimum_frames:
                                print(f"Video {self.filename} was too short, deleting...")
                                os.remove(self.filename)
                            recorded_frames = 0

                    if time.perf_counter() - timemark > 60:
                        self.ctx['__obj']['__log'].setLog(f"[INFO] {self.ctx_rtsp['camera']} is capturing correctly")
                        timemark = time.perf_counter()

                except Exception as e:
                    print(e)
                    time_inactivity += 0.5
                    time.sleep(0.5)
                    if time_inactivity > 5:
                        self.ctx['__obj']['__log'].setLog(f"[ERROR] {self.ctx_rtsp['camera']} is not capturing correctly, retrying...")
                        self.ctx_rtsp["capture"].release()
                        self.ctx_rtsp["capture"] = cv2.VideoCapture(self.ctx_rtsp["camera"], cv2.CAP_FFMPEG)
                        time_inactivity = 0
            else:
                width = self.ctx_rtsp["width"] or 640
                height = self.ctx_rtsp["height"] or 480
                self.frame = np.zeros((height, width, 3), np.uint8)
                time.sleep(1)
                self.ctx_rtsp["capture"].release()
                self.ctx_rtsp["capture"] = cv2.VideoCapture(self.ctx_rtsp["camera"], cv2.CAP_FFMPEG)
                time_inactivity = 0

        if self.ctx_rtsp['verbose']:
            self.ctx['__obj']['__log'].setLog('[INFO] Finished reading frames cleanly in [' + str(time.perf_counter() - timemark) + '] sec.')

    def get_frame(self):
        return self.frame.copy() if hasattr(self, 'frame') and self.frame is not None else None

    def startRecording(self):
        self.record_ongoing = True
        GP = self.ctx['__obj']['__global_procedures']
        timestamp = GP.getTodayString(mask="%Y_%m_%d-%I_%M_%S_%p")
        camera_name = self.ctx_rtsp['camera'].lower().replace(' ', '_')
        file = f"videos/{camera_name}-{timestamp}.mp4"
        dir = GP.createDirectory(dir=['records'], base=self.ctx_rtsp['folder'])
        self.filename = os.path.join(dir, file)
        self.ctx_rtsp['output_video'] = ffmpegcv.VideoWriter(
            self.filename,
            self.ctx_rtsp['codec'],
            self.ctx_rtsp['fps']
        )
        self.recording = True

    def stopRecording(self):
        self.recording = False
        self.allow_stop = True

    def release(self):
        self.active = False
        if self.recording:
            self.stopRecording()
        self.ctx_rtsp['capture'].release()
        cv2.destroyAllWindows()
        self.ctx['__obj']['__log'].setLog("[INFO] Finished releasing video capture and output")
