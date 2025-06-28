import subprocess
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from ngsildclient import Client

load_dotenv("./back/.env")

# Connect to Context Broker
host = os.getenv('HOST')
client = Client(hostname=host,port=1026)

# Initialize BlobServiceClient for Azure connection
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
account_url = os.getenv("AZURE_ACCOUNT_URL")
container_name = os.getenv("CONTAINER_NAME")
download_dir = "./back/Downloads"
blob_service_client = BlobServiceClient(account_url, credential=sas_token)

def run_script(script_path):
    try:
        # Ejecuta el script y espera a que termine
        subprocess.run(["python", script_path], check=True)
        print(f"Successfully executed {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}: {e}")

if __name__ == "__main__":
    print("=== main.py started ===")
    while True:
        ## Check if there are new videos to infer
        # Get all entities of type videoRecorded 
        entities = client.query(type="videoRecorded")
        # Check if there are any entities with inferred=false
        new_videos = [entity for entity in entities if entity['inferred']['value'] == False and entity['path']['value'] is not None]

        if new_videos:
            print(f"Found {len(new_videos)} new videos to process.")
        
            # Ruta para descargar los videos (solo si inferred=false)
            download_videos = "./back/video_downloader.py"

            # Ruta para procesar los videos con yolo y subirlos a la base de datos (todas las cámaras)
            process_videos = "./back/video_processing.py"

            # Ruta para el hacer spatial join (duración en galerias - solo cámaras 4 y 5)
            spatial_join_galeries = "./back/durations_spatial_join.py"

            # Ruta para detectar cruce de linea (solar hub - solo cámara 5)
            line_crossing = "./back/cross_product_line_crossing.py"

            # Ejecutar el script para descargar videos
            print("Downloading videos...")
            run_script(download_videos)

            # Ejecutar el script de procesamiento
            print("Processing video entities and inputting them in database...")
            run_script(process_videos)

            # Ejecutar el script de spatial join
            print("Perfoming spatial join...")
            run_script(spatial_join_galeries)

            # Ejecutar el script de spatial join
            print("Detecting line crossings...")
            run_script(line_crossing)

            # Esperar media hora antes de volver a verificar
            print("Waiting for 30 minutes before checking for new videos...")
            time.sleep(1800)  # 30 minutes in seconds
            
        else:
            print("No new videos to process. Waiting for 30 minutes before checking again...")
            time.sleep(1800)