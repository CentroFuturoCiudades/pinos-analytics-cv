from threading import Thread
import os
import time
import subprocess
import numpy as np
import ffmpeg
from Generic.Global.Borg import Borg
import cv2
from queue import Queue

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

        # on self.inactive_timeout seconds without receiving frames, the stream will be considered inactive
        self.inactive_timeout = 5
        
        self.visual = visualize
        self.active = True
        self.stream_active = False
        self.recording = False
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.frame_queue = Queue(maxsize=1)  # Queue to hold the latest frame
        self.process = None

        self.mutithreadingRead()

    def mutithreadingRead(self):
        self.ctx['__obj']['__log'].setLog('Starting multithreading')
        self.start_read_thread()
        self.cameraThread = Thread(target=self.update_queue)
        self.cameraThread.daemon = True
        self.cameraThread.start()
        if self.visual:
            self.show_video()
            
    def start_read_thread(self):
        self.ctx['__obj']['__log'].setLog(f"[INFO] Starting the read thread for {self.ctx_rtsp['camera']}")
        self.active = True
        self.streamThread = Thread(target=self.stream_update)
        self.streamThread.daemon = True
        self.streamThread.start()


    def stream_update(self):
        self.ctx['__obj']['__log'].setLog(f"[INFO] Starting the capturing process in the {self.ctx_rtsp['camera']}, height: {self.ctx_rtsp['height']}, width: {self.ctx_rtsp['width']}")
        width = self.ctx_rtsp["width"]
        height = self.ctx_rtsp["height"]
        self.process = (
            ffmpeg
            .input(self.rtsp_url, rtsp_transport='tcp')
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=5)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        frame_size = width * height * 3
        
        while self.active:
            try:
                in_bytes = self.process.stdout.read(frame_size)
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                self.frame_queue.put(frame, timeout=1)  # Put the frame in the queue
                self.stream_active = True
            except Exception as e:
                self.ctx['__obj']['__log'].setLog(f"[ERROR] [{self.ctx_rtsp['camera']}] Error processing frame: {e}")
                break
        
    def update_queue(self):
        self.ctx['__obj']['__log'].setLog(f"[INFO] Starting the update queue process in the {self.ctx_rtsp['camera']}")
        last_frame_time = time.time()
        while self.active:
            try:
                frame = self.frame_queue.get(timeout=5)  # Get the latest frame from the queue
                # print(f"frame updated on {self.ctx_rtsp['camera']}")
                if frame is not None:
                    self.frame = frame
                    last_frame_time = time.time()
            except Exception as e:
                self.ctx['__obj']['__log'].setLog(f"[ERROR] [{self.ctx_rtsp['camera']}] Error getting frame from queue: {e}")
            
            # Check if the stream is inactive
            if time.time() - last_frame_time > self.inactive_timeout:
                self.stream_active = False
                self.ctx['__obj']['__log'].setLog(f"[WARNING] Stream inactive for {self.inactive_timeout} seconds in {self.ctx_rtsp['camera']}")
                self.process.terminate()  # Terminate the ffmpeg process
                self.streamThread.join(timeout=10)  # Wait for the stream thread to finish
                self.start_read_thread()  # Restart the stream thread
                
        self.ctx['__obj']['__log'].setLog(f"[INFO] Stopping the update queue process in the {self.ctx_rtsp['camera']}")

    def get_frame(self):
        response = self.frame.copy() if self.frame is not None else None
        self.frame = None # Clear the frame to avoid returning same frame repeatedly -> and detect dead streams
        return response

    def show_video(self):
        self.videoThread = Thread(target=self.show_video_thread)
        self.videoThread.daemon = True
        self.videoThread.start()

    def show_video_thread(self):
        if self.ctx_rtsp["verbose"]:
            self.ctx['__obj']['__log'].setLog("[INFO] Showing video...")
        while self.active:
            cv2.imshow('frame', self.frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

    def startRecording(self):
        GP = self.ctx['__obj']['__global_procedures']
        file = self.ctx_rtsp['camera'] + "_" + GP.getTodayString("%Y_%m_%d-%I_%M_%S_%p") + '.mp4'
        
        # while file exists, append a number to make it unique
        i = 0
        while os.path.exists(os.path.join(self.ctx_rtsp['folder'], file)):
            file = file.replace('.mp4', f'_{i}.mp4')
            i += 1
        
        dir = GP.createDirectory(['records', GP.getTodayString("%Y_%m_%d"), self.ctx_rtsp['camera']], base=self.ctx_rtsp['folder'])
        self.filename = os.path.join(dir, file)
        self.ctx['__obj']['__log'].setLog(f"[INFO] Starting ffmpeg recording to {self.filename}")

        # Start FFmpeg without duration limit
        self.ffmpeg_proc = subprocess.Popen([
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", "30",  # 1 minute chunks as safety (adjust as needed)
            "-segment_format", "mp4",
            "-reset_timestamps", "1",
            "-movflags", "+faststart",
            "-strftime", "1",
            "-crf", "28",  # (lower is better quality, 28 is good for saving space) default is 23
            "-y", self.filename
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        self.recording_start_time = time.time()
        self.last_movement_time = time.time()  # Track last movement
        self.recording = True

    def stopRecording(self):
        if hasattr(self, 'ffmpeg_proc') and self.ffmpeg_proc:
            try:
                # Gracefully stop FFmpeg
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
        """Call this from MovementDetector when movement is detected"""
        self.last_movement_time = time.time()

    def shouldStopRecording(self, clip_duration=5):
        """Call this periodically from MovementDetector to check if should stop"""
        return (time.time() - self.last_movement_time) >= clip_duration
    
    def get_state(self):
        """Get the current state of the recorder"""
        if self.recording:
            return 'recording'
        elif self.stream_active:
            return 'active'
        else:
            return 'inactive'
