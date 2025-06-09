from collections import defaultdict
import json
import datetime
import cv2
from ultralytics import YOLO
import numpy as np
import os

VIDEO_NAME = 'camera1_2025_05_26-04_19_19_PM' # camera#_YYYY_MM_DD-HH_MM_SS
SAVE_VIDEO = True

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

count = 0

if SAVE_VIDEO:
    output_path = f'../../vids/test/{VIDEO_NAME}sigma_s=8_sigma_r=0.15.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame = cv2.detailEnhance(frame, sigma_s=8, sigma_r=0.15)
        #result = model.track(frame, persist=True, imgsz=(frame_height * 2, frame_width * 2), conf=0.4)[0] # Yolo processing in better image quality with imgsz=(frame_height, frame_width)
        result = model.track(frame, persist=True, imgsz=(frame_height, frame_width))[0] # Yolo processing in better image quality with imgsz=(frame_height, frame_width)
        #result = model.track(frame, persist=True)[0] #OLD
                    
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

                    #Print to test out
                    print(f"timestamp {timestamp_formatted}; detection_id: {track_id}, bbox: {json.dumps(box.cpu().numpy().tolist())}; skeleton: {json.dumps(keypoints[i].tolist())};")
                    count += 1

                if SAVE_VIDEO:
                    out.write(frame)
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
if SAVE_VIDEO:
    out.release()
cv2.destroyAllWindows()

if len(track_history) > 0:
    print("Keypoints found")

else:
    print("No keypoints found")
    # ENDED YOLO11 PROCESSING

print(f"Total detections: {count}")