import numpy as np
import cv2 as cv
from ultralytics import YOLO as yolo

model = yolo("models/drone-yolo26m.pt") #load a pretrained yolo model


#create a VideoCapture object (can access files OR camera (use 0 for camera, 1 for external camera, etc.))
cap = cv.VideoCapture("videos/test4.mp4") 

#make sure the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

#run while loop to read frames from the video
while cap.isOpened():
    success, frame = cap.read() #read a frame from the video
    
    if not success:
        print("Could not read frame (stream ended). Exiting...")
        break

    result = model.track( #add tracking to the model. provide the frame to track
        frame,
        persist = True,
        tracker = "bytetrack.yaml",
        imgsz = 704,
        conf = 0.20,
        verbose = False,
        device = 0,
        quantize = 16)[0] #run the model on the frame and get the results (detections)
    annotated_frame = result.plot()
    display_frame = cv.resize(annotated_frame, (1280, 720))
    cv.imshow("UAV Tracker", display_frame) #display the frame in a window

    if cv.waitKey(1) & 0xFF == ord('q'): #play the video (1ms frame switch) until 'q' is pressed
        break       

cap.release() #release the video capture object
cv.destroyAllWindows() #close all OpenCV windows