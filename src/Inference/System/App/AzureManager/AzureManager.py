import os
from datetime import datetime

from azure.storage.blob import BlobServiceClient
from System.util.date import date_from_filename
from System.util.file import file_name
from tqdm import tqdm

from Generic.Global.Borg import Borg


class AzureManager(Borg):
    def __init__(self, sas_token: str, verbose=False) -> None:
        """
        Class builder, initializes the AzureUploader with account URL and SAS token.

        Args:
            account_url: str
            sas_token: str
        Returns:
            [None]: None
        """
        super().__init__()
        self.ctx["__obj"]["__log"].setLog("Initializing AzureManager")

        account_url = self.ctx["__obj"]["__config"].get("azure").get("account_url")
        self.default_container = (
            self.ctx["__obj"]["__config"].get("azure").get("container")
        )
        self.blob_service_client = BlobServiceClient(account_url, credential=sas_token)
        self.verbose = verbose
        self.download_dir = (
            self.ctx["__obj"]["__config"].get("save_paths").get("videos")
        )

        self.ctx["__obj"]["__log"].setLog("AzureManager initialized")

    
    def download_videos_by_range(
        self, start_time: datetime, end_time: datetime, container_name: str = None
    ) -> list:
        container_name = (
            self.default_container if container_name is None else container_name
        )
        container_client = self.blob_service_client.get_container_client(container_name)

        blobs_list = container_client.list_blobs()

        # Save blobs corresponding to timestamp
        filtered_blobs = []
        counter = 0

        for blob in tqdm(blobs_list, desc="Filtering blobs"):
            counter += 1
            # Get the blob's last modified time
            try:
                video_date = date_from_filename(blob.name).timestamp()

                # Check if the last_modified time is within the specified time period
                if start_time.timestamp() <= video_date <= end_time.timestamp():
                    filtered_blobs.append(blob)
            except Exception as e:
                self.ctx["__obj"]["__log"].setLog(
                    f"Error while filtering blobs: {e}. Blob name: {blob.name}"
                )

        self.ctx["__obj"]["__log"].setLog(f"Total blobs: {counter}")
        video_paths = []

        for blob in tqdm(filtered_blobs, desc="Downloading videos"):
            blob_client = container_client.get_blob_client(blob)
            download_file_path = os.path.join(self.output_dir, blob.name)
            video_paths.append(download_file_path)
            os.makedirs(os.path.dirname(download_file_path), exist_ok=True)

            # Download the blob
            with open(download_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

            if self.verbose:
                self.ctx["__obj"]["__log"].setLog(
                    f"Blob '{blob.name}' has been downloaded to '{download_file_path}'."
                )

        return video_paths

    def download_video(
        self, azure_path: str | list[str], container_name: str = None
    ) -> str:
        return self.download_file(
            azure_path=azure_path,
            download_dir=self.download_dir,
            container_name=container_name,
        )

