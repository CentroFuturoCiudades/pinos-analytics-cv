from ultralytics import YOLO, checks, hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv

TIMESTAMP = '2025_05_21-10_02_35_AM' #modify according to video name in IOT-AGENT/records
DAY = '2025_05_21' #modify according to video name in IOT-AGENT/records

# Load the YOLO11 model
model = YOLO("yolo11l-pose.pt")

# Open the video file
video_path = f"IOT-Agent/records/{DAY}/camera1/{TIMESTAMP}.mp4"
cap = cv2.VideoCapture(video_path)


# Get frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # You can use 'MP4V' for .mp4 files
fps = 20.0  # Adjust as per your camera's FPS
out = cv2.VideoWriter(f'../vids/output_video_p_{TIMESTAMP}.avi', fourcc, fps, frame_size)

lowest_points = []

if not cap.isOpened():
    print("No se pudo abrir el stream.")

else:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("No se pudo leer el frame.")
            break

        results = model(frame)

        for result in results:
            kpts = result.keypoints.xy  #get x, y coordinates of pose points
            if kpts is not None:
                ##get the last & second-to last values of the list, and keep the one with the lowest y
                for person in kpts:
                    kpts_person = np.array(person)
                    y_coordinates = kpts_person[:, 1]
                    if len(y_coordinates) > 0:
                        max_index = np.argmax(y_coordinates)
                        max_point = kpts_person [max_index] #get point with biggest y-coordinate (furthest down on img)

                        x, y = max_point
                        if x != 0 and y != 0:
                            lowest_points.append(max_point)
                            print(max_point)
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

                    #for x, y in person:
                        #cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

            #print(kpts)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Lowest points")
print(lowest_points)

lowest_points = np.array(lowest_points)

# Save to csv file
csv_file = f"../csv/trajectory/trajectory_points_p_{TIMESTAMP}.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["x", "y"])  # Header
    for x, y in lowest_points:
        writer.writerow([x, y])

#plot trajectory
cam_img = cv2.imread("../imgs/cam01.jpeg")
cam_img  = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
img_height, img_width, _ = cam_img.shape

fig, ax = plt.subplots()
ax.imshow(cam_img)

for (x, y) in lowest_points:
    ax.plot(x, y, "bo")

plt.savefig(f'../imgs/plottedtrajectories_p{TIMESTAMP}.png')
plt.show()