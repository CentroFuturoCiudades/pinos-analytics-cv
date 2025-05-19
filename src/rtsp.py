from ultralytics import YOLO, checks, hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

#For Camera 3
"""video = cv2.VideoCapture('rtsp://100.108.97.81:8554/cam1')
homography_matrix = np.array([
       [-6.64359014e+00,  1.68386720e+01,  5.47738945e+03],
       [-9.01311337e-01,  6.57976459e+00,  2.06128382e+03],
       [-2.21459681e-03,  9.71791265e-03,  1.00000000e+00]
       ], dtype=np.float32)"""

#For Camera 5
video = cv2.VideoCapture('rtsp://100.108.97.81:8554/cam2')
homography_matrix = np.array([
       [ 1.88329052e+00,  2.56231035e+00, -6.96170799e+02],
       [-6.13642040e-01,  3.89654647e+00, -4.53682995e+02],
       [ 1.90218797e-04, -1.27126773e-04,  1.00000000e+00]
       ], dtype=np.float32)

model = YOLO("yolo11l-pose.pt")

lowest_points = []

birdseye_img = cv2.imread("../imgs/birdseye.jpg")
birdseye_img = cv2.cvtColor(birdseye_img, cv2.COLOR_BGR2RGB)

if not video.isOpened():
    print("No se pudo abrir el stream.")

else:
    ret, frame = video.read()
    results = model(frame)
    while True:
        ret, frame = video.read()
        
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

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("Lowest points")
print(lowest_points)

lowest_points = np.array(lowest_points)

lowest_points = lowest_points.reshape(-1, 1, 2)
# Transform the point using the homography matrix
transformed_points = cv2.perspectiveTransform(lowest_points, homography_matrix) # in birdseye view
transformed_points = transformed_points.reshape(-1, 2)

fig, ax = plt.subplots()
ax.imshow(birdseye_img)

for (x, y) in transformed_points:
    ax.plot(x, y, "bo")

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
plt.savefig(f'../imgs/plottedtrajectories{timestamp}.png')
plt.show()
video.release()