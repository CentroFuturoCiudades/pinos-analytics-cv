import os
import pandas as pd
from collections import defaultdict
import json
import cv2
import sqlalchemy as sa
import numpy as np
import datetime
import time
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from ngsildclient import Client, Entity, iso8601

# Connect to PostGIS URI
host = "100.85.126.64"
port = 5434
db = "oasis"
user = "admin"
password = "admin"
engine = sa.create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# Read the detectionsobserved table from PostGIS
#table = pd.read_sql("SELECT * FROM detectionsobserved", engine)
#print(table)

# Load the YOLO11 model
model = YOLO("yolo11x-pose.pt")

# Connect to Context Broker
client = Client(hostname="100.85.126.64",port=1026)
# Load environment variables
load_dotenv()

# Initialize BlobServiceClient for Azure connection
sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")
account_url = os.getenv("AZURE_ACCOUNT_URL")
container_name = "oasis"
download_dir = "./Downloads"
blob_service_client = BlobServiceClient(account_url, credential=sas_token)

# Get all entities of type videoRecorded 
entities = client.query(type="videoRecorded")

# For entities where inferred is false, access video
for entity in entities:
    if entity['inferred']['value'] == False and entity['path']['value'] is not None:
        blob_path = entity['path']['value']
        print(f"Accessing video at", {blob_path})

        # Download the video from Azure Blob Storage
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
        local_path = os.path.join(download_dir, blob_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the video to process locally
        with open(local_path, "wb") as f:
            data = blob_client.download_blob()
            f.write(data.readall())
            print(f"Downloaded video to {local_path}")

        # STARTING YOLO11 PROCESSING
        print(f"Processing video {blob_path}...")

        # Open the video file
        video_path = f"Downloads/{blob_path}"
        cap = cv2.VideoCapture(video_path)

        # Store the track history
        track_history = defaultdict(lambda: [])

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLO11 tracking on the frame, persisting tracks between frames
                result = model.track(frame, persist=True, imgsz=(frame_height, frame_width))[0] # Yolo processing in better image quality with imgsz=(frame_height, frame_width)
                    
                # Get the boxes and track IDs
                if result.boxes and result.boxes.id is not None:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Visualize the result on the frame
                    frame = result.plot()

                    # Plot the tracks
                    for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                        if hasattr(result, "keypoints") and result.keypoints is not None:
                            keypoints = result.keypoints.xy.cpu().numpy()
                            track = track_history[track_id]

                            # Get timestamp into datetime format
                            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                            filename = entity['path']['value']
                            if filename.endswith('.mp4'):
                                filename = filename[:-4]
                            split_filename = filename.split('_', 1)

                            if len(split_filename) > 1:
                                timestamp_str = split_filename[1]
                                try:
                                    timestamp_formatted = datetime.datetime.strptime(timestamp_str, "%Y_%m_%d-%I_%M_%S_%p")
                                except ValueError:
                                    try:
                                        timestamp_formatted = datetime.datetime.strptime(filename, "%Y_%m_%d-%I_%M_%S_%p")
                                    except ValueError:
                                        print(f"Could not parse timestamp from: {timestamp_str}")
                                        continue

                            # Calculate the timestamp for the current frame
                            timestamp_formatted += datetime.timedelta(seconds= int(frame_number / fps))

                            #Upload processed entity to PostGIS database
                            with engine.begin() as conn:
                                # Insert into detectionsobserved table if ID does not already exist
                                conn.execute(sa.text("""
                                    INSERT INTO detectionsobserved (id, video_path, timestamp, detection_id, bbox, skeleton, camera_number)
                                    VALUES (:id, :video_path, :timestamp, :detection_id, :bbox, :skeleton, :camera_number)
                                    ON CONFLICT (id) DO NOTHING
                                """), {
                                    "id": f"camera{entity['camera']['value']}_{track_id}_{timestamp_formatted}_{frame_number}",
                                    "video_path": entity['path']['value'],
                                    "timestamp": timestamp_formatted,
                                    "detection_id": track_id,
                                    "bbox": json.dumps(box.cpu().numpy().tolist()),
                                    "skeleton": json.dumps(keypoints[i].tolist()),
                                    "camera_number": entity['camera']['value']
                                })
                                row_uploaded = pd.read_sql(f"SELECT * FROM detectionsobserved WHERE id = 'camara{entity['camera']['value']}_{timestamp_formatted}_{track_id}'", conn)

                            print(f"Uploading {entity['id']} to PostGIS database.")
                    
                    # Display the annotated frame
                    #cv2.imshow("YOLO11 Tracking", frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

        if len(track_history) > 0:
            print("Keypoints found")

        else:
            print("No keypoints found")
        # ENDED YOLO11 PROCESSING

        # Mark inferred as true
        entity['inferred']['value'] = True
        print(f"Marked {entity['id']} as inferred.")
        print()

        # Delete video from local storage
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"Deleted local video file {local_path}.")
        else:
            print(f"Local video file {local_path} does not exist.")
        
        print("length track_history is ", len(track_history))
        # Update the entity in Context Broker
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.update(entity)
                print(f"Updated entity {entity['id']} in Context Broker.")
                break
            except Exception as e:
                print(f"Failed to update entity (attempt {attempt+1}): {e}")
                time.sleep(2)
        else:
                print(f"Could not update entity {entity['id']} after {max_retries} attempts.")
    
        #break #REMOVE TO PROCESS ALL VIDEOS

# Close the database connection
engine.dispose()
# Close the Context Broker client connection
client.close()
# Close the BlobServiceClient connection
blob_service_client.close()