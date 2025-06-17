from azure.storage.blob import BlobServiceClient
import os
from glob import glob
from dotenv import load_dotenv
import subprocess
from ngsildclient import Client, Entity, iso8601
# local imports
from Generic.Global.Borg import Borg

class Uploader(Borg):

    def __init__(self) -> None:
        """
        Class builder, initializes the Uploader with account URL and SAS token.
        
        Args: 
            account_url: str
            sas_token: str
        Returns:
            [None]: None
        """
        super().__init__()
        load_dotenv()  # Cargar las variables del .env
        account_url = os.getenv("AZURE_ACCOUNT_URL")
        sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
        
        if not account_url or not sas_token:
            raise ValueError("Azure credentials not found in environment variables")

        self.ctx['__obj']['__log'].setLog('Initializing Uploader from .env')
        self.blob_service_client = BlobServiceClient(account_url, credential=sas_token)
        self.client = Client(hostname="100.85.126.64",port=1026)        
        self.ctx['__obj']['__log'].setLog('Uploader initialized')

    def upload_video(self, local_file_name: str) -> None:
        """
        Upload a video file to Azure Blob Storage.
        
        Args: 
            local_file_name: str
        Returns:
            [None]: None
        """
        self.ctx['__obj']['__log'].setLog(f"Uploading video to Azure Storage: {local_file_name}")
        try:
            blob_name = os.path.basename(local_file_name)
            blob_client = self.blob_service_client.get_blob_client(container="oasis", blob=blob_name)

            self.ctx['__obj']['__log'].setLog(f"Uploading to Azure Storage as blob: {local_file_name}")

            with open(file=local_file_name, mode="rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            self.ctx['__obj']['__log'].setLog(f"Uploaded {local_file_name} to Azure Blob Storage.")
            #create entity
            try:
                id_camera = int(local_file_name[local_file_name.index('camera') + 6])
            except ValueError:
                self.ctx['__obj']['__log'].setLog(f'Failed getting id camera')
                return None
            # Create video entity
            video_entity = {
                "camera": id_camera,
                "path": blob_name,
            }
            # Create video entity
            self.create_videoRecorded_entity(video_entity)
        except Exception as e:
            self.ctx['__obj']['__log'].setLog(f"Error uploading {local_file_name} to Azure Blob Storage: {e}")
            return False
        return True

    def create_videoRecorded_entity(self, video: dict) -> dict:
        """
        Create a videoRecorded entity.

        Args:
            video: dict
        Returns:
            dict: Entity data
        """
        self.ctx['__obj']['__log'].setLog('Creating videoRecorded entity')
        dt = video["path"].split('.')[0]
        e = Entity("videoRecorded", f"camera{video['camera']}:{dt}")
        e.prop("camera", video["camera"])
        e.prop("path", video["path"])
        e.prop("inferred", False)
        self.client.upsert(e)
        self.ctx['__obj']['__log'].setLog('videoRecorded entity created')




    def loadProcess(self) -> None:
        """
        This method upload all .mp4 files and delete them from records folder.

        Args:
            None
        Returns:
            None
        """

        files = glob('records/**/*.mp4',recursive=True)
        self.ctx['__obj']['__log'].setLog(f'Loaded files: {files}')
        for f in files:
            self.ctx['__obj']['__log'].setLog(f'Uploading {f}')
            success = self.upload_video(f)
            if success:
                os.remove(f)
                self.ctx['__obj']['__log'].setLog(f'Deleted file {f}')
         
