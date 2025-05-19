from threading import Thread
import cv2
import os
import datetime
import ffmpegcv
import time
import imutils
import argparse
#Local imports
from Generic.Global.Borg import Borg
from System.App.Uploader.Uploader import Uploader
import System.App.RTSPRecorder.gstream_python as gs
import multiprocessing as mp
import numpy as np
#Director class
class RTSPRecorder(Borg):

    USE_GSTREAMER = False
    
    #Contextual generic objects
    __ctx = None

    #Configuration data
    __config = None

    def __init__(self, 
                 camera: str='camera1', 
                 folder: str='',
                 codec: str='h264',
                 width: int=None,
                 height: int=None,
                 fps: int=None,
                 verbose: bool=False,
                 visualize: bool=False):

        """
        Class builder, all the contextual configurations are charged from the base class (borg pattern) in a shared state
        Returns:
            [None]: None
        """
        #Setting contextual generic objects
        self.ctx = Borg._Borg__shared_state['ctx']        
        src = self.ctx['__obj']['__config'].get('rtsp')[camera]
        self.ctx['__obj']['__log'].setLog( f"[INFO] Reading from {src}" )
        self.ctx_rtsp = {   "src": src,
                            "camera": camera, 
                            "capture": cv2.VideoCapture ( src, cv2.CAP_FFMPEG ) if not self.USE_GSTREAMER else None,
                            "output_video": None,
                            "folder": folder,
                            "fps": fps,
                            "width": width,
                            "height": height,
                            "codec": codec,
                            "verbose": verbose
                        }
    
        #Gstream settings
        if self.USE_GSTREAMER:
            self.cam_queue = None
            self.stopbit = None
            self.active_track = False
            self.gstream_timeout = 60
            self.ctx['__obj']['__log'].setLog( f"[INFO] Using GStreamer for {src}" )
        
        self.visual = visualize
        # A video queue to start recording considering previous frames
        self.queued_seconds = 3
        self.minimum_frames = 30 #minimum frames recorded to not skip clip
        self.video_queue = []
        self.record_queue = []
        self.writing_video = False
        self.active = True
        self.recording = False
        self.record_ongoing = False
        self.allow_stop = False
        self.frame = np.zeros((self.ctx_rtsp["height"], self.ctx_rtsp["width"], 3), np.uint8)
        self.error_time_acum = 0
        if self.ctx_rtsp['fps'] is None:
            self.ctx_rtsp['fps'] = self.ctx_rtsp['capture'].get(cv2.CAP_PROP_FPS)
            
        self.multithreadingRead()
        self.uploader = Uploader()
        
        #Bye
        return None
    
    def multithreadingRead(self):
        """
        Multithreading reading of RTSP Streming of the IP cameras. 
        Args:
            [None]: None
                               
        Returns:
            [None]: None
        """
        self.ctx['__obj']['__log'].setLog('Starting multithreading')
        if self.USE_GSTREAMER:
            #start cam stream
            self.startGStream()
            self.waitForGStream()
        else:
            self.readStatus, frame = self.ctx_rtsp["capture"].read()
            while not self.readStatus:
                self.ctx['__obj']['__log'].setLog( "[ERROR] Failed to read from video stream, retrying...")
                self.recover_stream()
                self.readStatus, frame = self.ctx_rtsp["capture"].read()
        #Start the thread to read frames from the video stream
        self.cameraThread = Thread(target=self.update_queue, args=())
        self.cameraThread.daemon = True
        self.readStatus = False
        self.cameraThread.start()
        if self.visual:
            self.show_video()
        #Bye
        return None

    def resize(self,frame):
        #Resize frame to desired width/height
        if self.ctx_rtsp["width"] is not None or self.ctx_rtsp["height"] is not None:
            #If one of the dimensions is None, resize while maintaining aspect ratio
            rframe = imutils.resize(frame, 
                                        width=self.ctx_rtsp["width"], 
                                        height=self.ctx_rtsp["height"])
        else:
            rframe = frame
        
        return rframe

    def update(self):
        #Setting initial time mark for recording procedure
        timemark = time.perf_counter()
        #Starting log
        self.ctx['__obj']['__log'].setLog( f"[INFO] {self.ctx_rtsp['camera']}: Starting capture process" )
        # Read the next frame from the stream in a different thread
        time_inactivity = 0
        while self.active:
            if self.ctx_rtsp["capture"].isOpened():
                try:
                    #print("Reading frame")
                    (self.readStatus, frame) = self.ctx_rtsp["capture"].read()
                    #resize
                    self.frame = self.resize(frame)
                    if self.frame is None or not self.readStatus:
                        continue
                    if self.recording:
                        self.writing_video = True
                        self.ctx_rtsp['output_video'].write(self.frame)
                        self.writing_video = False
                    # Every one minute, print the camera is ok
                    if time.perf_counter() - timemark > 60:
                        self.ctx['__obj']['__log'].setLog( f"[INFO] {self.ctx_rtsp['camera']}: is capturing correcly" )
                        timemark = time.perf_counter()
                except Exception as e:
                    # one frame was missed, but the stream is probably still alive
                    time_inactivity += 0.5
                    time.sleep(0.5)
                    if time_inactivity > 5:
                        self.ctx['__obj']['__log'].setLog( f"[ERROR] {self.ctx_rtsp['camera']}: is not capturing correcly, retrying..." )
                        # close and reopen the stream
                        self.ctx_rtsp["capture"].release()
                        self.ctx_rtsp["capture"] = cv2.VideoCapture ( self.ctx_rtsp["camera"], cv2.CAP_FFMPEG )
                        time_inactivity = 0
                    pass
            else:
                # self.ctx['__obj']['__log'].setLog( f"[ERROR] {self.ctx_rtsp['camera']} is not capturing correcly, retrying..." )
                # send black image to stop any recording
                self.frame = np.zeros((self.ctx_rtsp["height"], self.ctx_rtsp["width"], 3), np.uint8)
                time.sleep(1)
                # close and reopen the stream
                self.ctx_rtsp["capture"].release()
                self.ctx_rtsp["capture"] = cv2.VideoCapture ( self.ctx_rtsp["camera"], cv2.CAP_FFMPEG )
                time_inactivity = 0
        if self.ctx_rtsp['verbose']:
            self.ctx['__obj']['__log'].setLog('[INFO] Finished reading frames cleanly in [' + str( time.perf_counter() - timemark ) + '] sec.')

    def update_queue(self):
        # queue frames for up to 5 seconds
        # queue frames for up to 5 seconds
        #Setting initial time mark for recording procedure
        timemark = time.perf_counter()
        self.recorder_start_time = time.perf_counter()
        #Starting log
        self.ctx['__obj']['__log'].setLog( f"[INFO] {self.ctx_rtsp['camera']}: Starting capture process" )
        # Read the next frame from the stream in a different thread
        time_inactivity = 0
        recorded_frames = 0
        
        wait_ms = int(1000/self.ctx_rtsp['fps'])
        prev_time_ms = time.perf_counter_ns() // 1000000
        
        error_time = 0
        had_error = False
        
        while self.active or self.record_ongoing:
            if (time.perf_counter_ns() // 1000000 - prev_time_ms > wait_ms) or self.USE_GSTREAMER:
                try:
                    frame = None
                    if self.USE_GSTREAMER:
                        if not self.cam_queue.empty():
                            cmd, val = self.cam_queue.get()
                            if cmd == gs.StreamCommands.FRAME:
                                if val is not None:
                                    frame = val
                        elif self.camProcess.is_stream_active() == False:
                            # print('Error reading frame, caught from gstreamer')
                            # display blank frame
                            self.frame = np.zeros((self.ctx_rtsp["height"], self.ctx_rtsp["width"], 3), np.uint8)
                            if not had_error:
                                self.ctx['__obj']['__log'].setLog( f"[ERROR] {self.ctx_rtsp['camera']}: Error reading from camera, started recovery..." )
                                error_time = time.time()
                                had_error = True
                            self.recoverGStream()
                            continue
                        else:
                            continue
                    else:
                        (self.readStatus, frame) = self.ctx_rtsp["capture"].read()
                        if not self.readStatus:
                            if not had_error:
                                self.ctx['__obj']['__log'].setLog( f"[ERROR] Error reading from {self.ctx_rtsp['camera']}, started recovery..." )
                                error_time = time.time()
                            had_error = True
                            # print("Error reading frame caught with readStatus")
                            self.recover_stream()
                            continue
                    if had_error:
                        self.ctx['__obj']['__log'].setLog( f"[ERROR] {self.ctx_rtsp['camera']}: Error reading from camera, started recovery..." )
                        self.error_time_acum += time.time() - error_time
                        had_error = False
                    #resize
                    self.frame = self.resize(frame)
                    self.video_queue.append(self.frame)
                    # remove the oldest frame if the queue is too long
                    if len(self.video_queue) > self.queued_seconds * self.ctx_rtsp['fps']:
                        self.video_queue.pop(0)
                    # append the frames to the record queue
                    if self.recording:
                        # use a new flag, if it is false start a list withcurrent queue, then, write each frame of the queue until it is empty. 
                        # Outside, use flag to know if it shpould append to this new queue, so if it should be written or not
                        if len(self.record_queue) == 0:
                            self.record_queue = self.video_queue.copy()
                        else:
                            self.record_queue.append(self.frame)
                    # write the frames to the video file
                    if self.record_ongoing:
                        if len(self.record_queue) > 0:
                            self.writing_video = True
                            record_frame = self.record_queue.pop(0)
                            self.ctx_rtsp['output_video'].write(record_frame)
                            recorded_frames += 1
                            self.writing_video = False
                        elif self.allow_stop:
                            self.ctx_rtsp['output_video'].release()
                            self.record_ongoing = False
                            recorded_frames = 0

                    # Every one minute, print the camera is ok
                    if time.perf_counter() - timemark > 60:
                        self.ctx['__obj']['__log'].setLog( f"[INFO] {self.ctx_rtsp['camera']}: is capturing correcly" )
                        timemark = time.perf_counter()
                        
                except Exception as e:
                    # An unexpected error occurred, try to recover the stream anyways
                    if not had_error:
                        self.ctx['__obj']['__log'].setLog( f"[ERROR] {self.ctx_rtsp['camera']}: Error reading from camera, started recovery..." )
                        error_time = time.time()
                    had_error = True
                    # print("Error reading frame")
                    if self.USE_GSTREAMER:
                        self.recoverGStream()
                    else:
                        self.recover_stream()
                    pass
        if self.ctx_rtsp['verbose']:
            self.ctx['__obj']['__log'].setLog(f'[INFO] {self.ctx_rtsp["camera"]}: Finished reading frames cleanly in [{time.perf_counter() - timemark}] sec.')
            self.camProcess.terminate()
        pass
    
    def recover_stream(self):
        self.ctx_rtsp["capture"].release()
        self.frame = np.zeros((self.ctx_rtsp["height"], self.ctx_rtsp["width"], 3), np.uint8)
        time.sleep(5)
        self.ctx_rtsp["capture"] = cv2.VideoCapture ( self.ctx_rtsp["src"], cv2.CAP_FFMPEG )
    
    def stopGStream(self):
        # print('in stopCamStream')

        if self.stopbit is not None:
            self.stopbit.set()
            while not self.cam_queue.empty():
                try:
                    _ = self.cam_queue.get()
                except:
                    break
                self.cam_queue.close()
            print("Waiting for join")
            self.camProcess.join()
            print("Joined")
    
    def startGStream(self):
        #set  queue size
        self.cam_queue = mp.Queue(maxsize=100)
        self.stopbit = mp.Event()
        self.camProcess = gs.StreamCapture(self.ctx_rtsp["src"],
                             self.stopbit,
                             self.cam_queue,
                            self.ctx_rtsp['fps'],
                            verbose=False)
        self.camProcess.start()
    
    def recoverGStream(self):
        self.stopGStream()
        time.sleep(5)
        self.startGStream()
        # print("Waiting for stream to restart...")
        self.waitForGStream()
        
    def waitForGStream(self):
        startTime = time.time()
        while self.camProcess.is_stream_active() == False and time.time() - startTime < self.gstream_timeout:
            time.sleep(2)
        if time.time() - startTime >= self.gstream_timeout:
            return False
        return True
    
    def get_frame(self):
        # Return the most recent frame (a numpy array) from the video stream
        frame = self.frame.copy()
        return frame

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        
        # Press Q on keyboard to stop frame show
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
    
    def show_video(self):
        # wait at most 5 seconds for self.readStatus to be True, dont use time.sleep
        self.videoThread = Thread(target=self.show_video_thread, args=())
        #self.videoThread.daemon = True
        self.videoThread.start()

    def show_video_thread(self):
        # Display video from the stream
        wait_ms = int(1000/self.ctx_rtsp['fps'])
        prev_time_ms = time.perf_counter_ns() // 1000000
        if self.ctx_rtsp["verbose"]:
            self.ctx['__obj']['__log'].setLog(f"[INFO] {self.ctx_rtsp['camera']}: Showing video")
        while self.active:
            if time.perf_counter_ns() // 1000000 - prev_time_ms > wait_ms:
                cv2.imshow(self.ctx_rtsp['camera'], self.frame)
                # Press Q on keyboard to stop video show
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.capture.release()
                    cv2.destroyAllWindows()
                    break
                prev_time_ms = time.perf_counter_ns() // 1000000
        

    def startRecording(self):
        self.record_ongoing = True
        #Build directory path and filename
        GP = self.ctx['__obj']['__global_procedures']
        file = GP.getTodayString(mask = "%Y_%m_%d-%I_%M_%S_%p") + '.mp4'
        dir = GP.createDirectory(dir = ['records', 
                                        GP.getTodayString("%Y_%m_%d"), 
                                        self.ctx_rtsp['camera']
                                       ],
                                 base = self.ctx_rtsp['folder']
                                )
        self.uploader.createDir(self.ctx_rtsp['camera'])  
        self.filename = dir + file
        #Build videowriter object
        self.ctx_rtsp['output_video'] = ffmpegcv.VideoWriter(self.filename,
                                                    self.ctx_rtsp['codec'],
                                                    self.ctx_rtsp['fps'])
        self.recording = True
    
    def stopRecording(self):
        # Stop recording frames
        self.recording = False
        self.allow_stop = True
        #while self.writing_video:
         #   pass
        #self.ctx_rtsp['output_video'].release()

    
    def release(self):
        # Release the video capture and output
        self.ctx['__obj']['__log'].setLog(f"[INFO] {self.ctx_rtsp['camera']}: Releasing video capture and output")
        if self.recording:
            print(f"[INFO] {self.ctx_rtsp['camera']}: Stopping recording due to release request")
            self.stopRecording()
            while self.record_ongoing:
                pass
        print(f"Recording stopped on {self.ctx_rtsp['camera']} from release request")
        self.active = False
        self.ctx['__obj']['__log'].setLog(f"[INFO] {self.ctx_rtsp['camera']}: Finished releasing video capture and output")
        self.ctx['__obj']['__log'].setLog(f"[INFO] {self.ctx_rtsp['camera']} was active for {time.perf_counter() - self.recorder_start_time} seconds with offline time of {self.error_time_acum} seconds")



