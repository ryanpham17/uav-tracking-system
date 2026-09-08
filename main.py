import numpy as np
import cv2 as cv
from ultralytics import YOLO as yolo

model = yolo("models/drone-yolo26m.pt") #load a pretrained yolo model
#model = yolo("models/yolo26n.pt") #use to test with cars

#create a VideoCapture object (can access files OR camera (use 0 for camera, 1 for external camera, etc.))
cap = cv.VideoCapture("videos/test.mp4") 

#make sure the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv.CAP_PROP_FPS) #get the frames per second of the video

track_history = {} #dictionary to store the track history of each object
telemetry = {} #dictionary to store the telemetry data of each object

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

    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist() #get the track IDs of the detected objects
        boxes = result.boxes.xywh.cpu().tolist() #get the bounding boxes of the detected objects

        #update the track history and telemetry data for each detected object
        for box, track_id in zip(boxes, track_ids):
            x, y, width, height = box #unpack the bounding box coordinates (x, y, width, height)

            if track_id not in track_history:
                track_history[track_id] = [] #initialize the track history for this object
                telemetry[track_id] = {} #initialize the telemetry data for this object

            track_history[track_id].append((int(x), int(y)))

            #calculate the distance moved by the object since the last frame
            if len(track_history[track_id]) >= 2: 
                previous_x, previous_y = track_history[track_id][-2]
                current_x, current_y = track_history[track_id][-1]

                dx = current_x - previous_x
                dy = current_y - previous_y

                distance = np.sqrt(dx**2 + dy**2)

                #eventually change for when detections are missing. need to use actual frame/time difference
                speed_px_per_sec = distance * fps #calculate speed in pixels per second

                #store the telemetry data for this object (updates every frame)
                telemetry[track_id] = {
                    "x": current_x,
                    "y": current_y,
                    "dx": dx,
                    "dy": dy,
                    "distance_px": distance,
                    "speed_px_per_sec": speed_px_per_sec
                }

                #print(f"Track ID: {track_id}, Telemetry: {telemetry[track_id]}")
                
            if len(track_history[track_id]) > 50:
                track_history[track_id].pop(0)

    annotated_frame = result.plot()

    for track_id, points in track_history.items(): #.items() lets you access key, value
        if len(points) > 1: #only draw the trajectory if there are at least 2 points
            points_array= np.array(points, dtype = np.int32) #convert the list of points to a numpy array
            cv.polylines( #draw the trajectory of the object on the frame
                annotated_frame,
                [points_array],
                isClosed = False,
                color = (255, 0, 0), #OpenCV uses BGR not RBG
                thickness = 5
            )

    display_frame = cv.resize(annotated_frame, (1280, 720))
    cv.imshow("UAV Tracker", display_frame) #display the frame in a window

    if cv.waitKey(1) & 0xFF == ord('q'): #play the video (1ms frame switch) until 'q' is pressed
        break       

cap.release() #release the video capture object
cv.destroyAllWindows() #close all OpenCV windows