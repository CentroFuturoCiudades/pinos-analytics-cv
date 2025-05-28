import os
import pandas as pd
from ngsildclient import Client, Entity, iso8601
from sqlalchemy import create_engine
from azure.storage.blob import BlobServiceClient

# Connect to PostGIS URI
host = "100.85.126.64"
port = 5434
db = "oasis"
user = "admin"
password = "admin"
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# Read the detectionsobserved table from PostGIS
table = pd.read_sql("SELECT * FROM detectionsobserved", engine)
print(table)

# Connect to Context Broker
client = Client(hostname="100.85.126.64",port=1026)

# Initialize BlobServiceClient for Azure connection
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
account_url = os.getenv("AZURE_ACCOUNT_URL")
container_name = "oasis"
download_dir = "./Inference/Downloads"
blob_service_client = BlobServiceClient(account_url, credential=sas_token)


# Get all entities of type videoRecorded - TO DO: remove [-10:]
entities = client.query(type="videoRecorded")[-10:]

# For entities where inferred is false, print the path
for entity in entities:
    if entity['inferred']['value'] == False:
        blob_path = entity['path']['value']
        print(f"Accessing video at", {path})

        # Download the video from Azure Blob Storage
        blob_service_client.get_blob_client(container=container_name, blob=blob_path)
        local_path = os.path.join(download_dir, path=path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "wb") as f:
            data = blob_client.download_blob()
            f.write(data.readall())
        print(f"Downloaded video to {local_path}")

        #TO DO: process entities, upload to db
        print(f"Uploading {entity['id']} to PostGIS database.")

        # Mark inferred as true
        #entity['inferred']['value'] = True
        print(f"Marked {entity['id']} as inferred.")
        print()