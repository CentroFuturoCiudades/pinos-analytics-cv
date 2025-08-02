import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

account_url = os.getenv("AZURE_ACCOUNT_URL")
container_name = os.getenv("CONTAINER_NAME")
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
blob_service_client = BlobServiceClient(account_url, credential=sas_token)

def download_video(video_name, download_dir="./videos"):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=video_name)
    local_path = os.path.join(download_dir, video_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download the video to process locally
    with open(local_path, "wb") as f:
        data = blob_client.download_blob()
        f.write(data.readall())
        print(f"Downloaded video to {local_path}")
