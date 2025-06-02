from collections import defaultdict
import json
import datetime
import cv2
from ultralytics import YOLO
import numpy as np
import os

VIDEO_NAME = 'camera1_2025_05_27-10_30_49_AM' # camera#_YYYY_MM_DD-HH_MM_SS

# Load the YOLO11 model
model = YOLO("yolo11x-pose.pt")

print(f"Processing video {VIDEO_NAME}...")
# Open the video file
video_path = f"Downloads/{VIDEO_NAME}.mp4"
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
        result = model.track(frame, persist=True)[0]
                    
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

                    # Get entity['inferred']['value'] into datetime format
                    timestamp_entity = "2025_05_27-10_30_49_AM"
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    timestamp_formatted = datetime.datetime.strptime(timestamp_entity, "%Y_%m_%d-%I_%M_%S_%p")

                    # Calculate the timestamp for the current frame
                    timestamp_formatted += datetime.timedelta(seconds=(frame_number % int(fps)))

                    #Upload processed entity to PostGIS database
                    print(f"timestamp {timestamp_formatted}; detection_id: {track_id}, bbox: {json.dumps(box.cpu().numpy().tolist())}; skeleton: {json.dumps(keypoints[i].tolist())};")
                    
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

