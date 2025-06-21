from multiprocessing import Process, Queue, Value, Array
import os
import time
import subprocess
import numpy as np
import ffmpeg
from Generic.Global.Borg import Borg
import cv2
import threading

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
        self.stream_active = Value('i', 0)  # Shared boolean (0=False, 1=True)
        self.recording = False
        
        # Create shared array for frame data
        self.frame_size = width * height * 3
        self.shared_frame = Array('B', self.frame_size)
        self.frame_ready = Value('i', 0)
        self.frame_lock = threading.Lock()
        
        # Local frame for external access
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        self.frame_queue = Queue(maxsize=1)  # Queue to hold the latest frame
        self.process = None
        self.stream_process = None
        self.camera_process = None
        self.video_process = None

        self.mutithreadingRead()

    def mutithreadingRead(self):
        self.ctx['__obj']['__log'].setLog('Starting multiprocessing')
        self.camera_process = Process(target=self.update_queue)
        self.camera_process.daemon = False
        self.camera_process.start()
        
        # Start a local thread to update the main process frame
        self.frame_update_thread = threading.Thread(target=self.update_local_frame, daemon=False)
        self.frame_update_thread.start()
        
        if self.visual:
            self.show_video()

    def update_local_frame(self):
        """Updates the local frame from shared memory for external access"""
        while self.active:
            try:
                if self.frame_ready.value:
                    with self.frame_lock:
                        # Copy from shared memory to local frame
                        frame_data = np.frombuffer(self.shared_frame.get_obj(), dtype=np.uint8)
                        self.frame = frame_data.reshape((self.ctx_rtsp['height'], self.ctx_rtsp['width'], 3)).copy()
                        self.frame_ready.value = 0 
                time.sleep(0.01) 
            except Exception as e:
                self.ctx['__obj']['__log'].setLog(f"[ERROR] Error updating local frame: {e}")
                time.sleep(0.1)

    def start_read_process(self):
        self.ctx['__obj']['__log'].setLog(f"[INFO] Starting the read process for {self.ctx_rtsp['camera']}")
        self.active = True
        self.stream_process = Process(target=self.stream_update)
        self.stream_process.daemon = False
        self.stream_process.start()

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
                if len(in_bytes) != frame_size:
                    break
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame if exists
                    except:
                        pass
                self.frame_queue.put(frame, timeout=1)  # Put the frame in the queue
                self.stream_active.value = 1
            except Exception as e:
                self.ctx['__obj']['__log'].setLog(f"[ERROR] [{self.ctx_rtsp['camera']}] Error processing frame: {e}")
                break
        
        if self.process:
            self.process.terminate()

    def update_queue(self):
        self.ctx['__obj']['__log'].setLog(f"[INFO] Starting the update queue process in the {self.ctx_rtsp['camera']}")
        last_frame_time = time.time()
        self.start_read_process()
        while self.active:
            try:
                frame = self.frame_queue.get(timeout=5)  # Get the latest frame from the queue
                if frame is not None:
                    # Update shared memory with the new frame
                    with self.frame_lock:
                        frame_flat = frame.flatten()
                        self.shared_frame[:len(frame_flat)] = frame_flat
                        self.frame_ready.value = 1  # Signal that frame is ready
                    
                    last_frame_time = time.time()
                    self.stream_active.value = 1
            except Exception as e:
                self.ctx['__obj']['__log'].setLog(f"[ERROR] [{self.ctx_rtsp['camera']}] Error getting frame from queue: {e}")
            
            # Check if the stream is inactive
            if time.time() - last_frame_time > self.inactive_timeout:
                self.stream_active.value = 0
                self.ctx['__obj']['__log'].setLog(f"[WARNING] Stream inactive for {self.inactive_timeout} seconds in {self.ctx_rtsp['camera']}")
                
                # Terminate the ffmpeg process
                if self.process:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except:
                        self.process.kill()
                        
                # Terminate the stream process forcefully
                if self.stream_process and self.stream_process.is_alive():
                    self.stream_process.terminate()
                    self.stream_process.join(timeout=5)
                    if self.stream_process.is_alive():
                        self.stream_process.kill()
                        self.stream_process.join()
                
                # Restart the stream process
                time.sleep(2)  # Wait before restarting
                self.start_read_process()
                last_frame_time = time.time()  # Reset timer
                
        self.ctx['__obj']['__log'].setLog(f"[INFO] Stopping the update queue process in the {self.ctx_rtsp['camera']}")

    def get_frame(self):
        """Returns a copy of the current frame for external use"""
        with self.frame_lock:
            response = self.frame.copy() if self.frame is not None else None
            self.frame = None
            return response

    def is_stream_active(self):
        """Returns whether the stream is currently active"""
        return bool(self.stream_active.value)

    def show_video(self):
        self.video_process = Process(target=self.show_video_process)
        self.video_process.daemon = True
        self.video_process.start()

    def show_video_process(self):
        if self.ctx_rtsp["verbose"]:
            self.ctx['__obj']['__log'].setLog("[INFO] Showing video...")
        while self.active:
            if self.frame is not None:
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
                self.ffmpeg_proc.wait(timeout=3)
            except:
                self.ffmpeg_proc.kill()
            finally:
                self.recording = False
                self.ctx['__obj']['__log'].setLog(f"[INFO] Finished recording to {self.filename}")

    def release(self):
        self.active = False
        if self.recording:
            self.stopRecording()
        
        # Wait for frame update thread to finish
        if hasattr(self, 'frame_update_thread') and self.frame_update_thread.is_alive():
            self.frame_update_thread.join(timeout=2)
        
        # Terminate all processes
        processes = [self.stream_process, self.camera_process, self.video_process]
        for proc in processes:
            if proc and proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
        
        # Terminate ffmpeg process
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
        
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
