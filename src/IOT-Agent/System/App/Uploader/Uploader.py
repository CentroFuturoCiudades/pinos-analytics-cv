import time
import boto3
from glob import glob
import os
import json
import uuid
import asyncio
from datetime import datetime
import requests
import subprocess
#Local imports
from Generic.Global.Borg import Borg
from System.App.Scrambling.Scrambling import Scrambling

#Director class
class Uploader(Borg):

    #Contextual generic objects
    __ctx = None

    #Configuration data
    __config = None

    #-----------------------------------------------------------------------------------------------------------------------------
    def __init__( self) -> None:
        """
        Class builder, all the contextual configurations are charged from the base class (borg pattern) in a shared state
        Returns:
            [None]: None
        """
        #Setting contextual generic objects
        self.ctx = Borg._Borg__shared_state['ctx']
        self.aws = self.ctx['__obj']['__config'].get('aws')
        session = boto3.Session(aws_access_key_id= self.aws['access_key_id'],
                               aws_secret_access_key= self.aws['secret_access_key'],
                               region_name= self.aws['region_name']
                              )
        self.s3 = session.resource('s3')
        self.bucket = self.s3.Bucket(self.aws['bucket'])
        self.s3_client = boto3.client('s3',
                        aws_access_key_id= self.aws['access_key_id'],
                        aws_secret_access_key= self.aws['secret_access_key'],
                        )
        self.API_URL = self.ctx['__obj']['__config'].get('backend')['url']
        self.CTX_HOSTNAME = self.ctx['__obj']['__config'].get('ctx_broker')['hostname']
        self.CTX_PORT = self.ctx['__obj']['__config'].get('ctx_broker')['port']
        self.DATE_FORMAT = '%Y_%m_%d-%I_%M_%S_%p'
        #Bye
        return None
    #---------------------------------------------------------------------------------------------------------------------------------
    def createDir(self, camera: str = 'camera1') -> None:
        """
        This method construct a directory path in S3.

        Args:
            camera [str]
        Returns:
            None
        """
        GP = self.ctx['__obj']['__global_procedures']
        self.path = 'records/' + GP.getTodayString("%Y_%m_%d") + f'/{camera}/'
        exist = False
        #verify if directory path exist
        for obj in self.bucket.objects.all():
            if obj.key == self.path:
                exist = True
        #create directory path in bucket
        if not exist:
            self.bucket.put_object(Key=self.path)
            #validate the creation of dir
            for obj in self.bucket.objects.all():
                if obj.key == self.path:
                    self.ctx['__obj']['__log'].setLog(f'The directory {self.path} has been created successfully')
        else:
            self.ctx['__obj']['__log'].setLog(f'The directory {self.path} already exist')

        #bye
        return None
    

    def upFile(self, file_path: str = None) -> None:
        """
        This method upload a file in a directory path in S3.

        Args:
            path [list]
        Returns:
            None
        """
        if file_path == None:
            self.ctx['__obj']['__log'].setLog(f'You need to specify the file to upload')
            return
        base_file = os.path.basename(file_path)
        self.path = os.path.dirname(file_path[file_path.index('records'):]) + '/'
        #verify if the file already exist
        exist = False
        for obj in self.s3_client.list_objects(Bucket=self.aws['bucket'])['Contents']:
            if obj['Key'] == self.path + base_file:
                exist = True
        #upload the file
        if not exist:
            self.s3_client.upload_file(file_path, self.aws['bucket'], self.path + base_file)
            #validate that the file has been already uploaded
            for obj in self.s3_client.list_objects(Bucket=self.aws['bucket'])['Contents']:
                if obj['Key'] == self.path + base_file:
                    self.ctx['__obj']['__log'].setLog(f'The file {base_file} has been uploaded successfully')
                    #create entity
                    try:
                        id_camera = int(file_path[file_path.index('camera') + 6])
                    except ValueError:
                        self.ctx['__obj']['__log'].setLog(f'Failed getting id camera')
                        return None
                    #asyncio.run(self.create_entity(id_camera, self.path + base_file))
                    self.create_entity_ctx(id_camera, self.path + base_file)
        else:
            self.ctx['__obj']['__log'].setLog(f'The file {base_file} already exist')
        #bye
        return None
    
    async def create_entity(self, cam_id: int, file: str) -> None:
        """
        This method calls an endpoint API to create entity of captured video in context broker.

        """
        # Create camera entity in API
        camera_data = dict(id=cam_id, bucket="tec-crowd-counting", file=file)
        response =  requests.post(f"{self.API_URL}/camera_footage", json=camera_data)
        camera_entity = response.json()
        self.ctx['__obj']['__log'].setLog(f'The following entity has been created: {camera_entity}')
    
    def create_entity_ctx(self, cam_id: int, file: str) -> None:
        """
        This method create entity in context broker
        """
        id = uuid.uuid4()
        date = datetime.strptime(os.path.basename(file.split('.')[0]), self.DATE_FORMAT) 
        timestamp = datetime.timestamp(date)
        payload={
        "id": f'urn:ngsi-ld:cameraObserved:{cam_id}:' + str(id) ,
        "type": "cameraObserved",
        'bucket': {
            'type': 'Property', 
            'value': 'tec-crowd-counting'
        },
        'file': {
            'type': 'Property', 
            'value': file
        },
        'infered': {
            'type': 'Property', 
            'value': False
        }, 
        'timestamp': {
            'type': 'Property', 
            'value': timestamp}
        }
        response = requests.post(url=f'http://{self.CTX_HOSTNAME}:{self.CTX_PORT}/ngsi-ld/v1/entities', headers={
            "content-type": "application/json"},  data=json.dumps(payload))
        if response.ok:
            self.ctx['__obj']['__log'].setLog('The entity has been created')
        else:
            self.ctx['__obj']['__log'].setLog('The entity creation failed')

    def check_video_length(self, video_path: str):
        try:
            result = subprocess.check_output(f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{video_path}"',
                                            shell=True).decode()
            fields = json.loads(result)['streams'][0]
            #print(fields)
            duration = fields['duration']
        except Exception as e:
            print("Error reading file")
            print(e)
            duration = 0
        return float(duration)

    def loadProcess(self) -> None:
        """
        This method upload all .enc files and delete them from records folder.

        Args:
            None
        Returns:
            None
        """

        files = glob('records/**/*.mp4',recursive=True)
        self.ctx['__obj']['__log'].setLog(f'Loaded files: {files}')
        for f in files:
            if self.check_video_length(f) < 3:
                self.ctx['__obj']['__log'].setLog(f'The file {f} is too short or unreadable')
                os.remove(f)
                continue
            self.ctx['__obj']['__log'].setLog(f'Encrypting record as {f}.enc')
            Scrambling.encryptFile(f)
            self.ctx['__obj']['__log'].setLog(f'Uploading {f}.enc')
            self.upFile(f + '.enc')
            #os.remove(f)
            self.ctx['__obj']['__log'].setLog(f'Deleted file {f}')
         
