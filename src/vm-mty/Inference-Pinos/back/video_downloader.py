import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from ngsildclient import Client

if __name__ == "__main__":
    load_dotenv()

    # Connect to Context Broker
    host = os.getenv('HOST')
    client = Client(hostname=host,port=1026)

    # Initialize BlobServiceClient for Azure connection
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
    account_url = os.getenv("AZURE_ACCOUNT_URL")
    container_name = os.getenv("CONTAINER_NAME")
    download_dir = "./back/Downloads"
    blob_service_client = BlobServiceClient(account_url, credential=sas_token)

    # Get all entities of type videoRecorded 
    entities = client.query(type="videoRecorded")

    # For entities where inferred is false, access video
    for entity in entities:
        if entity['inferred']['value'] == False and entity['path']['value'] is not None:
            blob_path = entity['path']['value']
            print(f"Downloading video at", {blob_path})

            # Download the video from Azure Blob Storage
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
            local_path = os.path.join(download_dir, blob_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download the video to process locally
            with open(local_path, "wb") as f:
                data = blob_client.download_blob()
                f.write(data.readall())
                print(f"Downloaded video to {local_path}")
        
            #break #REMOVE TO PROCESS ALL VIDEOS

    # Close the BlobServiceClient connection
    blob_service_client.close()