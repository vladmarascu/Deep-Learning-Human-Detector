# Created by Vlad Marascu for GDP ADD02 Group 2020
# 04/06/2020

# Script to be run upon entering the house, uses frontal camera as webcam input, returns bool variable 1 if PERSON is found

import numpy as np
import cv2

# List of class labels MobileNet+SSD framework was trained to detect
# Trained by chuanqi305: trained on COCO dataset, fine-tuned on PASCAL VOC, reaches a 72.7% mAP

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# Ignored classes set, everything except PEOPLE
IGNORE = set(["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"])

# Load pre-trained model from the same folder using the OpenCV dnn function (Ref: chuanqi305)
print("[INFO] loading model...")
network = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# Initialize video stream using OpenCV
print("[INFO] starting video stream...")
capture = cv2.VideoCapture(0)

# Loop over each frame from the video stream
while True:
    grabbed, frame = capture.read()  # Grab each frame
    # Determine the frames shape, extract width and height for later
    (h, w) = frame.shape[:2]
    # Transform to 4D blob (resize, crop from center, substract mean values, scale values by scalefactor)
    blob_4d = cv2.dnn.blobFromImage(cv2.resize(frame, (315, 315)),
                                    0.007843, (315, 315), 115.15)

    # Feed-forward the blob through the SSD+Mobilenet network and obtain the detections, our "targets"
    network.setInput(blob_4d)
    targets = network.forward()

    # Loop over each detection: determine where and what objects are and if label+box are drawn, depending on confidence value selected
    for i in np.arange(0, targets.shape[2]):
        # Extract the probability (confidence) associated with the detection (targets=detections)
        probability = targets[0, 0, i, 2]
        # Ensure probability is greater than the threshold (0.2), can be varied
        if probability > 0.2:
            # Index of the class label from 'targets'
            index = int(targets[0, 0, i, 1])
            # If the class label is in the set of ignored classes, skip detection
            if CLASSES[index] in IGNORE:
                continue
            # Find x,y start/end coords of the bounding box for each object as: X_start, Y_start, X_end, Y_end (start is top left, end is bottom right in pixel coords)
            bounding_box = targets[0, 0, i, 3:7] * np.array([w, h, w, h])
            (X_start, Y_start, X_end, Y_end) = bounding_box.astype("int")

            # Create label name for display
            label = "person"

            # Create bool variable that is 1 if humans are detected, 0 if no humans are derected (REQUIRED OUTPUT)

            varbool = 0
            if label == "person":
                varbool = 1
            else:
                varbool = 0
            # Draw bounding rectangle using the start/end coords extracted
            cv2.rectangle(frame, (X_start, Y_start), (X_end, Y_end),
                          (255, 0, 0), 2)

            # Display the label of the class (person) if detected
            cv2.putText(frame, label + str(probability*100), (X_start, Y_start - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Display bool variable (1 for proceed/ drop camera in vicinity)
            cv2.putText(frame, "Proceed:"+str(varbool), (X_start, Y_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the output frame as a videostream
    cv2.imshow("Frame", frame)
    # Print bool variable in terminal
    # print(str(varbool))
    # Break upon pressing 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release frames upon breaking/stop
cv2.destroyAllWindows()
capture.release()
