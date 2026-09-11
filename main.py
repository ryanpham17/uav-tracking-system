import numpy as np
import cv2 as cv
from ultralytics import YOLO as yolo

#model = yolo("models/drone-yolo26m.pt") #load a pretrained yolo model
model = yolo("models/yolo26n.pt") #use to test with cars

#create a VideoCapture object (can access files OR camera (use 0 for camera, 1 for external camera, etc.))
cap = cv.VideoCapture("videos/testroad.mp4") 

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv.CAP_PROP_FPS) #get the frames per second of the video

track_history = {} #store the track history of each object (key = track_id, value = list of (x, y) coordinates)
telemetry = {} #store the telemetry data of each object
last_seen = {} #store the last seen frame of each object (key = track_id, value = last seen frame)
last_position = {} #store the last position of each object (key = track_id, value = (x, y) coordinates)
speed_history = {} #store the speed history of each object (key = track_id, value = recent speed measurements)

current_frame = 0
max_missing_frames = 30 #max frames an object can be missing before it is considered lost

#run while loop to read frames from the video (frame by frame)
while cap.isOpened():
    success, frame = cap.read() #read a frame from the video
    
    if not success:
        print("Could not read frame (stream ended). Exiting...")
        break

    current_frame += 1

    result = model.track( #run the model on the frame and get the results (detections). track too
        frame,
        persist = True,
        tracker = "bytetrack.yaml",
        imgsz = 704,
        conf = 0.20,
        verbose = False,
        device = 0,
        quantize = 16)[0] 

    #work through every tracked object in the curr frame
    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist() #get the track IDs/bounding boxes of the detected objects
        boxes = result.boxes.xywh.cpu().tolist() #use xywh to get the center

        #update the track history and telemetry data for each detected object
        for box, track_id in zip(boxes, track_ids):
            x, y, width, height = box #unpack the bounding box coordinates (x, y, width, height)
            current_x = int(x)
            current_y = int(y)
            last_seen[track_id] = current_frame #update the last seen frame for each object in the curr frame

            if track_id not in track_history:
                track_history[track_id] = [] #initialize the track history for this object
                telemetry[track_id] = {} #initialize the telemetry data for this object
                speed_history[track_id] = [] #initialize the speed history for this object

            track_history[track_id].append((current_x, current_y))

            #calculate the distance moved by the object since the last frame
            if track_id in last_position: 
                previous_x, previous_y, previous_frame = last_position[track_id]

                dx = current_x - previous_x
                dy = current_y - previous_y

                distance = np.sqrt(dx**2 + dy**2)

                frame_difference = current_frame - previous_frame #used so program knows object was unseen for more than 1 frame

                if frame_difference > 0:
                    time_difference = frame_difference / fps
                    speed_px_per_sec = distance / time_difference
                    speed_history[track_id].append(speed_px_per_sec)

                    if len(speed_history[track_id]) > 5:
                        speed_history[track_id].pop(0) #keep only the last 5 speed measurements

                    smoothed_speed = np.mean(speed_history[track_id]) #calculate the average speed over the last 5 measurements

                    #store the telemetry data for this object (updates every frame)
                    telemetry[track_id] = {
                        "x": current_x,
                        "y": current_y,
                        "dx": dx,
                        "dy": dy,
                        "distance_px": distance,
                        "frame_difference": frame_difference,
                        "time_difference_sec": time_difference,
                        "speed_px_per_sec": speed_px_per_sec,
                        "smoothed_speed_px_per_sec": smoothed_speed
                    }

                    #print(f"Track ID: {track_id}, Telemetry: {telemetry[track_id]}")
            
            last_position[track_id] = (current_x, current_y, current_frame) #update the last position of this object
                
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

    #check for stale tracks to remove tracking
    stale_tracks = []

    for track_id, last_frame in last_seen.items():
        if current_frame - last_frame > max_missing_frames:
            stale_tracks.append(track_id)

    #pop the stale tracks from the dictionaries
    for track_id in stale_tracks:
        #print(f"Removing stale track: {track_id}")
        track_history.pop(track_id, None)
        telemetry.pop(track_id, None)
        last_seen.pop(track_id, None)
        last_position.pop(track_id, None)
        speed_history.pop(track_id, None)
        
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

cap.release()
cv.destroyAllWindows()