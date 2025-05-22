from collections import defaultdict

import cv2
from ultralytics import YOLO
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

TIMESTAMP = 'ADD_HERE' #modify according to video name in IOT-AGENT/records
DAY = 'ADD_HERE' #modify according to video name in IOT-AGENT/records

# Load the YOLO11 model
model = YOLO("yolo11l-pose.pt")

# Open the video file
video_path = f"IOT-Agent/records/{DAY}/camera1/{TIMESTAMP}.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object
output_path = f'../vids/output_trajectory_{TIMESTAMP}.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path , fourcc, fps, frame_size)

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
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            
            out.write(frame)
        
        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
out.release()

#print(track_history)

# Save to csv file
csv_file = f"../csv/trajectory/trajectory_points_{TIMESTAMP}.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["track_id", "x", "y"]) # Header

    for track_id, points in track_history.items():
        for x, y in points:
            writer.writerow([track_id, x, y])


#plot trajectory
cam_img = cv2.imread("../imgs/cam01.jpeg")
cam_img  = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
img_height, img_width, _ = cam_img.shape

fig, ax = plt.subplots()
ax.imshow(cam_img)

# Generate colors for each id
track_ids = sorted(track_history.keys())
cmap = cm.get_cmap("tab20", len(track_ids))
track_colors = {track_id: cmap(i) for i, track_id in enumerate(track_ids)}

# Plot trajectories
for track_id, points in track_history.items():
    if points:
        x_vals, y_vals = zip(*points)
        ax.plot(x_vals, y_vals, color=track_colors[track_id], label=f"ID {track_id}")

# Optional: Add legend
ax.legend(loc='upper right')
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)  # Flip y-axis to match image orientation

# Save and show
plt.axis('on')
plt.savefig(f'../imgs/plottedtrajectories_{TIMESTAMP}.png')
plt.show()