# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import math
import timeit
import time

KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7),
    (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)]
width = 640
height = 640

# Download the model from TF Hub.
model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/3')
movenet = model.signatures['serving_default']

# Threshold for
threshold = .3

# Loads video source (0 is for main webcam)
video_source = 2
cap = cv2.VideoCapture(video_source)

# Checks errors while opening the Video Capture
if not cap.isOpened():
    print('Error loading video')
    quit()

success, img = cap.read()

if not success:
    print('Error reding frame')
    quit()

y, x, _ = img.shape

while success:

    #속도 측정 시작 시점
    start_t = timeit.default_timer()

    # A frame of video or an image, represented as an int32 tensor of shape: 256x256x3. Channels order: RGB with values in [0, 255].
    tf_img = cv2.resize(img, (256, 256))
    tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
    tf_img = np.asarray(tf_img)
    tf_img = np.expand_dims(tf_img, axis=0)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf_img, dtype=tf.int32)

    # Run model inference.
    outputs = movenet(image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']
    points = []
    # iterate through keypoints
    i = 0
    for k in keypoints[0, 0, :, :]:
        # Converts to numpy array
        k = k.numpy()

        # Checks confidence for keypoint
        if k[2] > threshold:
            # The first two channels of the last dimension represents the yx coordinates (normalized to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints
            yc = int(k[0] * y)
            xc = int(k[1] * x)
            points.append((xc, yc))

            # Draws a circle on the image for each keypoint
            cv2.circle(img, (xc, yc), 2, (0, 255, 0), 5)
            cv2.putText(img, "{}".format(i), (int(xc), int(yc)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        else :
            points.append(None)
        i += 1



    #왼쪽 팔꿈치 각도 계산
    if points[5] != None and points[7] != None and points[9] != None:
        angle = math.degrees(math.atan2(points[9][1]-points[7][1], points[9][0]-points[7][0]) - math.atan2(points[5][1]-points[7][1], points[5][0]-points[7][0]))
        if angle < 0:
            angle += 360
        cv2.putText(img, str(int(angle)), (10, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))
        cv2.circle(img, (points[9][0]-100, points[9][1]), 2, (0, 0, 255), 5)

        Aangle = math.degrees(math.atan2(points[7][1] - points[9][1], points[7][0] - points[9][0]) - math.atan2(points[9][1]-points[9][1],points[9][0]-100-points[9][0]))
        if Aangle < 0:
            Aangle += 360
        cv2.putText(img, str(int(Aangle)), (10, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255))
        print(angle, Aangle)

    # 종료 시점
    terminate_t = timeit.default_timer()
    FPS = int(1. / (terminate_t - start_t))
    cv2.putText(img, str(int(FPS)), (10, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))

    # Shows image
    cv2.imshow('Movenet', img)
    # Waits for the next frame, checks if q was pressed to quit
    if cv2.waitKey(1) == ord("q"):
        break

    #time.sleep(2)
    # Reads next frame
    success, img = cap.read()

cap.release()