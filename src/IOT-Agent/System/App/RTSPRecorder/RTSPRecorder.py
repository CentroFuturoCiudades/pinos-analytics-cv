from threading import Thread, Lock
import os
import time
import subprocess
import numpy as np
import ffmpeg
from Generic.Global.Borg import Borg
import cv2

class RTSPRecorder(Borg):
    __ctx = None
    __config = None

    def __init__(self, camera='camera1', folder='', width=640, height=480, verbose=False, visualize=False):
        self.ctx = Borg._Borg__shared_state['ctx']
        self.rtsp_url = self.ctx['__obj']['__config'].get('rtsp')[camera]
        self.ctx['__obj']['__log'].setLog(f"[INFO] Reading from {self.rtsp_url}")

        self.ctx_rtsp = {
            "camera": camera,
            "folder": folder,
            "width": width,
            "height": height,
            "verbose": verbose
        }

        self.visual = visualize
        self.active = True
        self.recording = False
        self.lock = Lock()
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)

        self.mutithreadingRead()

    def mutithreadingRead(self):
        self.ctx['__obj']['__log'].setLog('Starting multithreading')
        self.cameraThread = Thread(target=self.update_queue)
        self.cameraThread.daemon = True
        self.cameraThread.start()
        if self.visual:
            self.show_video()

    def update_queue(self):
        self.ctx['__obj']['__log'].setLog(f"[INFO] Starting the capturing process in the {self.ctx_rtsp['camera']}")
        width = self.ctx_rtsp["width"]
        height = self.ctx_rtsp["height"]
        process = (
            ffmpeg
            .input(self.rtsp_url, rtsp_transport='tcp')
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        # Drain stderr to avoid blocking
        def drain_stderr(proc):
            for _ in proc.stderr:
                pass
        Thread(target=drain_stderr, args=(process,), daemon=True).start()

        while self.active:
            try:
                in_bytes = process.stdout.read(width * height * 3)
                if not in_bytes:
                    time.sleep(0.1)
                    continue
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                with self.lock:
                    self.frame = frame
            except Exception as e:
                self.ctx['__obj']['__log'].setLog(f"[ERROR] Exception in update_queue: {e}")
                time.sleep(1)

    def get_frame(self):
        with self.lock:
            return self.frame.copy()

    def show_video(self):
        self.videoThread = Thread(target=self.show_video_thread)
        self.videoThread.daemon = True
        self.videoThread.start()

    def show_video_thread(self):
        if self.ctx_rtsp["verbose"]:
            self.ctx['__obj']['__log'].setLog("[INFO] Showing video...")
        while self.active:
            with self.lock:
                cv2.imshow('frame', self.frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

    def startRecording(self):
        GP = self.ctx['__obj']['__global_procedures']
        file = GP.getTodayString("%Y_%m_%d-%I_%M_%S_%p") + '.mp4'
        dir = GP.createDirectory(['records', GP.getTodayString("%Y_%m_%d"), self.ctx_rtsp['camera']], base=self.ctx_rtsp['folder'])
        self.filename = os.path.join(dir, file)
        self.ctx['__obj']['__log'].setLog(f"[INFO] Starting ffmpeg recording to {self.filename}")

        self.ffmpeg_proc = subprocess.Popen([
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", "300",
            "-segment_format", "mp4",
            "-reset_timestamps", "1",
            "-movflags", "+faststart",
            "-strftime", "1",
            "-y", self.filename
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.recording_start_time = time.time()
        self.last_movement_time = time.time()
        self.recording = True

    def stopRecording(self):
        if hasattr(self, 'ffmpeg_proc') and self.ffmpeg_proc:
            try:
                self.ffmpeg_proc.stdin.write(b'q\n')
                self.ffmpeg_proc.stdin.flush()
                self.ffmpeg_proc.wait(timeout=5)
            except:
                self.ffmpeg_proc.kill()
            finally:
                self.recording = False
                self.ctx['__obj']['__log'].setLog(f"[INFO] Finished recording to {self.filename}")

    def release(self):
        self.active = False
        if self.recording:
            self.stopRecording()
        cv2.destroyAllWindows()
        self.ctx['__obj']['__log'].setLog("[INFO] Finished releasing video capture and output")

    def updateMovementTime(self):
        self.last_movement_time = time.time()

    def shouldStopRecording(self, clip_duration=5):
        return (time.time() - self.last_movement_time) >= clip_duration
