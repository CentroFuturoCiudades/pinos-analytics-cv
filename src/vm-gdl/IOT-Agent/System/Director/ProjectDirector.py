import datetime
import os
import time
from glob import glob
import ultralytics
import numpy as np
import asyncio

#Generic director
from Generic.Director.GenericProjectDirector import GenericProjectDirector

#Local imports
from System.App.Uploader.Uploader import Uploader
from System.App.MovementDetector.MovementDetector import MovementDetector, yolov8_warmup
from System.App.CameraMonitor.CameraMonitor import CameraMonitor

#Director class
class ProjectDirector( GenericProjectDirector ):

    #-----------------------------------------------------------------------------------------------------------------------------
    def __init__( self, ):
        """
        Class builder, all the contextual configurations are charged from the base class (borg pattern) in a shared state
        Returns:
            [None]: None
        """
        super().__init__(
                {
                    '__project': {
                        '__name': 'iot-agent',
                        '__label': 'iot-agent',
                    },
                }
            )
        
        

    
    #-----------------------------------------------------------------------------------------------------------------------------
    def __play( self, what, value_a, value_b ):
        """
        Main API starter objects flux
        """


        #Initial procedure log
        self.ctx['__obj']['__log'].setLog( 'Iniciando ...' )
        self.ctx['__obj']['__log'].setDebug( self.ctx ) 
        uploader = Uploader()
        
        self.camera_monitor = CameraMonitor(
            port=19800)
        self.camera_monitor.start()
        self.ctx['__obj']['__log'].setLog(f"Camera Monitor is running on {self.camera_monitor.get_url()}")
        # uploader.loadProcess()
        
        # loading model
        self.model = ultralytics.YOLO("yolo11x.pt")
        prevtime = time.time()
        yolov8_warmup(model=self.model, repetitions=10, verbose=False)
        self.ctx['__obj']['__log'].setLog(f"Model loaded in {time.time() - prevtime} seconds")

        #Step 01: Calling videoprocedures
        # Starting all objects
        sources=['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6'] # List of camera sources, can be IP cameras or local files
        movement_detectors = {}
        camera_states = {}
        TIME_TO_UPLOAD = 5 # Upload every 5 minutes
        TIME_TO_RECORD = 120 # Be active 120 minutes

        for src in sources:
            self.ctx['__obj']['__log'].setLog('Starting {}'.format(src))
            movement_detectors[src] = MovementDetector(
                camera=src,
                model=self.model,
                visualize=False,  # Solo si quieres ver el video
                verbose=True,
                clip_duration=5,
                time_between_detections=1
                )
            camera_states[src] = 'inactive'
        # Record for TIME_TO_RECORD seconds
        self.ctx['__obj']['__log'].setLog('Recording for {} minutes'.format(TIME_TO_RECORD))
        for src in sources:
            movement_detectors[src].start_inference()
        endTime = datetime.datetime.now() + datetime.timedelta(minutes=TIME_TO_RECORD)
        update_web_time = time.time()
        while datetime.datetime.now() < endTime: ###change to run indefinitely
            try:
                if time.time() - update_web_time > 10:
                    # Update camera states every 5 seconds
                    for src in sources:
                        camera_states[src] = movement_detectors[src].get_state()
                    self.camera_monitor.update_camera_states(camera_states)
                    update_web_time = time.time()
            except KeyboardInterrupt:
                print("Exit through keyboard interrupt")
                break
            time.sleep(0.1)  # Sleep to avoid busy waiting

        for src in sources:
             # close all
            movement_detectors[src].stop()
            self.ctx['__obj']['__log'].setLog('Stopped {}'.format(src))
        
        self.ctx['__obj']['__log'].setLog('Finished demo')
        #Bye
        return None

    
    #-----------------------------------------------------------------------------------------------------------------------------
    def setFlux( self, argv ):
        """
        Main API starter objects multiprocessing
        """
        #Main argument
        try:
            what = argv[1]
        except:
            what = None
        #Complementary argument A
        try:
            value_a = argv[2]
        except:
            value_a = None
        #Complementary argument B
        try:
            value_b = argv[3]
        except:
            value_b = None
        #Regular conciliation procedure?
        if (
            what is None or
            what == '-d' 
        ):
            """
                * Examples that you can run on console/shell:
                    a) [ python main.py -d yyyy-mm-dd yyyy-mm-dd ]
                    b) [ python main.py -d yyyy-mm-dd ]
                    c) [ python main.py ] --> This will be TODAY date
            """
            #Playing main conciliation
            self.__play(
                (
                    '-d' if what is None else what 
                ), 
                value_a, 
                value_b 
            )
        #Invalid argument?
        else:
            #Invalid execution argument log
            self.ctx['__obj']['__log'].setLog( 'Argumento de execucion [' + str( what ) + '] invalido' )
        #Goodbye
        return None
    
    #-----------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def go( argv ):
        """
        Main API starting flux
        """
        ProjectDirector().setFlux( argv )
