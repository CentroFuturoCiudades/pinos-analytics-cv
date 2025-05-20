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
from System.App.RTSPRecorder.RTSPRecorder import RTSPRecorder
from System.App.Scrambling.Scrambling import Scrambling
from System.App.Uploader.Uploader import Uploader
from System.App.MovementDetector.MovementDetector import MovementDetector, yolov8_warmup
from System.App.TestBorg.TestBorg import TestBorg
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
        # Parámetros
        CAMERA_ID = 'camera1'
        DURATION_SECONDS = 120  # Duración de monitoreo
        # loading model
        self.model = ultralytics.YOLO("yolov8n.pt")
        prevtime = time.time()
        yolov8_warmup(model=self.model, repetitions=10, verbose=False)
        self.ctx['__obj']['__log'].setLog(f"Model loaded in {time.time() - prevtime} seconds")

        # Instancia del detector
        detector = MovementDetector(
            camera=CAMERA_ID,
            model=self.model,
            visualize=False,  # Solo si quieres ver el video
            verbose=True,
            clip_duration=5,
            time_between_detections=1
        )

        # Iniciar detección
        detector.start_inference()

        # Esperar cierto tiempo
        time.sleep(DURATION_SECONDS)

        # Finalizar proceso
        detector.stop()
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
