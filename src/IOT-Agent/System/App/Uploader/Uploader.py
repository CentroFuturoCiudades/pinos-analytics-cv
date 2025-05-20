from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv
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
        except Exception as e:
            self.ctx['__obj']['__log'].setLog(f"Error uploading {local_file_name} to Azure Blob Storage: {e}")
