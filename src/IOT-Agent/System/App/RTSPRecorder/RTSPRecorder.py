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
import numpy as np
#Director class
class RTSPRecorder(Borg):

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
        self.ctx_rtsp = { "camera": camera,
                          "capture": cv2.VideoCapture ( src, cv2.CAP_FFMPEG ),
                          "output_video": None,
                          "folder": folder,
                          "fps": fps,
                          "width": width,
                          "height": height,
                          "codec": codec,
                          "verbose": verbose
                        }
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
        if self.ctx_rtsp['fps'] is None:
            self.ctx_rtsp['fps'] = self.ctx_rtsp['capture'].get(cv2.CAP_PROP_FPS)
            
            
        self.mutithreadingRead()
        self.uploader = Uploader()
        
        

        #Bye
        return None
    
    def mutithreadingRead(self):
        """
        Multithreading reading of RTSP Streming of the IP cameras. 
        Args:
            [None]: None
                               
        Returns:
            [None]: None
        """
        self.ctx['__obj']['__log'].setLog('Starting multithreading')
        self.readStatus, frame = self.ctx_rtsp["capture"].read()
        if not self.readStatus:
            self.ctx['__obj']['__log'].setLog( "[ERROR] Failed to read from video stream")
            return
        #resize
        self.frame = self.resize(frame)
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
        self.ctx['__obj']['__log'].setLog( f"[INFO] Starting the capturing process in the {self.ctx_rtsp['camera']}" )
        # Read the next frame from the stream in a different thread
        time_inactivity = 0
        while self.active:
            if self.ctx_rtsp["capture"].isOpened():
                try:
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
                        self.ctx['__obj']['__log'].setLog( f"[INFO] {self.ctx_rtsp['camera']} is capturing correcly" )
                        timemark = time.perf_counter()
                except Exception as e:
                    # one frame was missed, but the stream is probably still alive
                    time_inactivity += 0.5
                    time.sleep(0.5)
                    if time_inactivity > 5:
                        self.ctx['__obj']['__log'].setLog( f"[ERROR] {self.ctx_rtsp['camera']} is not capturing correcly, retrying..." )
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
        #Starting log
        self.ctx['__obj']['__log'].setLog( f"[INFO] Starting the capturing process in the {self.ctx_rtsp['camera']}" )
        # Read the next frame from the stream in a different thread
        time_inactivity = 0
        recorded_frames = 0
        while self.active:
            if self.ctx_rtsp["capture"].isOpened():
                try:
                    (self.readStatus, frame) = self.ctx_rtsp["capture"].read()
                    #resize
                    self.frame = self.resize(frame)
                    self.video_queue.append(self.frame)
                    if len(self.video_queue) > self.queued_seconds * self.ctx_rtsp['fps']:
                        self.video_queue.pop(0)
                    if self.frame is None or not self.readStatus:
                        continue
                    if self.recording:
                        # use a new flag, if it is false start a list withcurrent queue, then, write each frame of the queue until it is empty. 
                        # Outside, use flag to know if it shpould append to this new queue, so if it should be written or not
                        if len(self.record_queue) == 0:
                            self.record_queue = self.video_queue.copy()
                        else:
                            self.record_queue.append(self.frame)
                    if self.record_ongoing:
                        if len(self.record_queue) > 0:
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

                    # Every one minute, print the camera is ok
                    if time.perf_counter() - timemark > 60:
                        self.ctx['__obj']['__log'].setLog( f"[INFO] {self.ctx_rtsp['camera']} is capturing correcly" )
                        timemark = time.perf_counter()
                except Exception as e:
                    # one frame was missed, but the stream is probably still alive
                    print(e)
                    time_inactivity += 0.5
                    time.sleep(0.5)
                    if time_inactivity > 5:
                        self.ctx['__obj']['__log'].setLog( f"[ERROR] {self.ctx_rtsp['camera']} is not capturing correcly, retrying..." )
                        # close and reopen the stream
                        self.ctx_rtsp["capture"].release()
                        self.ctx_rtsp["capture"] = cv2.VideoCapture ( self.ctx_rtsp["camera"], cv2.CAP_FFMPEG )
                        time_inactivity = 0
                    pass
            else:
                # print( f"[ERROR] {self.ctx_rtsp['camera']} is not capturing correcly, retrying..." )
                # send black image to stop any recording
                self.frame = np.zeros((self.ctx_rtsp["height"], self.ctx_rtsp["width"], 3), np.uint8)
                time.sleep(1)
                # close and reopen the stream
                self.ctx_rtsp["capture"].release()
                self.ctx_rtsp["capture"] = cv2.VideoCapture ( self.ctx_rtsp["camera"], cv2.CAP_FFMPEG )
                time_inactivity = 0
        if self.ctx_rtsp['verbose']:
            self.ctx['__obj']['__log'].setLog('[INFO] Finished reading frames cleanly in [' + str( time.perf_counter() - timemark ) + '] sec.')
        pass
    
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
        startTime = time.time()
        while not self.readStatus:
            if time.time() - startTime > 5:
                self.ctx['__obj']['__log'].setLog("[ERROR] Failed to show video stream")
                return
        self.videoThread = Thread(target=self.show_video_thread, args=())
        self.videoThread.daemon = True
        self.videoThread.start()

    def show_video_thread(self):
        # Display video from the stream
        if self.ctx_rtsp["verbose"]:
            self.ctx['__obj']['__log'].setLog("[INFO] Showing video...")
        while self.readStatus and self.active:
            cv2.imshow('frame', self.frame)
            # Press Q on keyboard to stop video show
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.capture.release()
                cv2.destroyAllWindows()
                break

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
        self.active = False
        if self.recording:
            self.stopRecording()
        self.ctx_rtsp['capture'].release()
        cv2.destroyAllWindows()
        self.ctx['__obj']['__log'].setLog("[INFO] Finished releasing video capture and output")


