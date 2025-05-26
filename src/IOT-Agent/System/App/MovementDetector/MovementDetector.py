import threading
import cv2
import time
import datetime
import numpy as np
from Generic.Global.Borg import Borg
from System.App.RTSPRecorder.RTSPRecorder import RTSPRecorder

def yolov8_warmup(model, repetitions=1, verbose=False):
    # Warmup model
    warmupFrame = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(repetitions):
        model.predict(source=warmupFrame, verbose=verbose)


class MovementDetector(Borg):
    def __init__(self, camera='camera1', model=None, treshold=0.002, 
                 time_between_detections=1, clip_duration=5, folder='', 
                 width=640, height=480, verbose=False, visualize=False):

        self.ctx = Borg._Borg__shared_state['ctx']
        src = self.ctx['__obj']['__config'].get('rtsp')[camera]
        self.ctx_mov = {
            "src": src, "src_name": camera, "model": model, "treshold": treshold,
            "time_between_detections": time_between_detections, "clip_duration": clip_duration,
            "folder": folder, "width": width, "height": height,
            "verbose": verbose, "visualize": visualize
        }

        self.TIME_BETWEEN_DETECTIONS = time_between_detections
        self.CLIP_MIN_DURATION = clip_duration
        self.BLUR_INTENSITY = 5
        self.DIFFERENCE_THRESHOLD = 0.001
        self.PIXEL_TRESHOLD = 30

        self.RECORDING = 1
        self.NOT_RECORDING = 0
        self.curr_state = self.NOT_RECORDING
        self.recording_start_time = 0

        self.src = src
        self.src_name = camera
        self.treshold = treshold
        self.folder = folder
        self.width = width
        self.height = height
        self.verbose = verbose
        self.visualize = visualize
        self.model = model

        self.recorder = RTSPRecorder(camera=camera, folder=folder, width=width, height=height, verbose=verbose, visualize=visualize)
        self.run_active = False
        self.ctx['__obj']['__log'].setLog(f"[INFO] Initialized movement detection object for {self.src_name}")

    def start(self):
        self.run_active = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def start_inference(self):
        if self.ctx_mov["model"] is None:
            self.ctx['__obj']['__log'].setLog("[ERROR] No model specified")
            return
        self.run_active = True
        self.thread = threading.Thread(target=self.run_inference)
        self.thread.start()

    def run(self):
        prev_frame = cv2.GaussianBlur(self.recorder.get_frame(), (self.BLUR_INTENSITY, self.BLUR_INTENSITY), 0)
        last_frame_time = time.time()
        while self.run_active:
            if time.time() - last_frame_time < self.TIME_BETWEEN_DETECTIONS:
                time.sleep(0.01)
                continue
            last_frame_time = time.time()
            frame = self.recorder.get_frame()
            if frame is None:
                self.ctx['__obj']['__log'].setLog(f"[WARNING] {self.src_name} : Empty frame, skipping")
                continue
            frame_blur = cv2.GaussianBlur(frame, (self.BLUR_INTENSITY, self.BLUR_INTENSITY), 0)
            diff = cv2.absdiff(frame_blur, prev_frame)
            diff[diff < self.PIXEL_TRESHOLD] = 0
            percent_diff = np.sum(diff) / (diff.size * 255)
            if percent_diff > self.DIFFERENCE_THRESHOLD:
                if self.curr_state == self.NOT_RECORDING:
                    self.recorder.startRecording()
                    self.recording_start_time = time.time()
                    self.curr_state = self.RECORDING
            elif self.curr_state == self.RECORDING and time.time() - self.recording_start_time > self.CLIP_MIN_DURATION:
                self.recorder.stopRecording()
                self.curr_state = self.NOT_RECORDING
            prev_frame = frame_blur

    def run_inference(self):
        def extract_detections(outs, threshold):
            boxes, confidences, classids = [], [], []
            for out in outs:
                for box in out.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if box.conf[0] >= threshold:
                        boxes.append([x1, y1, x2 - x1, y2 - y1])
                        confidences.append(float(box.conf[0]))
                        classids.append(int(box.cls[0]))
            return boxes, confidences, classids

        last_frame_time = time.time()
        while self.run_active:
            if time.time() - last_frame_time < self.TIME_BETWEEN_DETECTIONS:
                time.sleep(0.1)
                continue
            last_frame_time = time.time()
            frame = self.recorder.get_frame()
            if frame is None:
                continue
            detections = self.model.predict(source=frame, verbose=True, classes=0)
            boxes, _, classids = extract_detections(detections, 0.4)
            if 0 in classids:
                if self.curr_state == self.NOT_RECORDING:
                    self.recorder.startRecording()
                    self.recording_start_time = time.time()
                    self.curr_state = self.RECORDING
                    self.ctx['__obj']['__log'].setLog(f"[INFO] {self.src_name} : Detected movement, starting recording after detect {len(classids)} persons")
                else:
                    self.ctx['__obj']['__log'].setLog(f"[INFO] {self.src_name} : Detected movement, recording")
                    continue
            elif self.curr_state == self.RECORDING and time.time() - self.recording_start_time > self.CLIP_MIN_DURATION:
                self.recorder.stopRecording()
                self.ctx['__obj']['__log'].setLog(f"[INFO] {self.src_name} : Stopped recording after {time.time() - self.recording_start_time } seconds")
                self.curr_state = self.NOT_RECORDING

    def stop(self):
        self.run_active = False
        if self.curr_state == self.RECORDING:
            self.recorder.stopRecording()
        self.recorder.release()

    def is_recording(self):
        return self.curr_state == self.RECORDING
