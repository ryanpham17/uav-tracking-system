import numpy as np
import cv2 as cv
from ultralytics import YOLO as yolo

def create_kalman_filter(x, y, dt):
    kf = cv.KalmanFilter(4, 2)

    #define the state transition matrix (A) and measurement matrix (H)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0], #[x, y, vx, vy]
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    #define the measurement matrix (H)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03 #process noise covariance: How much do I distrust my assumption that the object moves at constant velocity?
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0 #measurement noise covariance: How much do I distrust my assumption that the object moves at constant velocity?

    #initialize the state vector (x, y, dx, dy) with the initial position and zero velocity
    kf.statePost = np.array([
        [x],
        [y],
        [0],
        [0]
    ], dtype=np.float32)

    return kf

#predict the future position of an object given its current position, velocity, and time ahead
def predict_future_position(x, y, vx, vy, time_ahead):
    future_x = x + vx * time_ahead
    future_y = y + vy * time_ahead

    return int(future_x), int(future_y)

model = yolo("models/drone-yolo26m.pt") #drone model
#model = yolo("models/yolo26n.pt") #use to test with cars

#create a VideoCapture object (can access files OR camera (use 0 for camera, 1 for external camera, etc.))
cap = cv.VideoCapture("videos/test.mp4") 

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv.CAP_PROP_FPS) #get the frames per second of the video
dt = 1 / fps #calculate the time difference between frames. delta time

track_history = {} #store the track history of each object (key = track_id, value = list of (x, y) coordinates)
telemetry = {} #store the telemetry data of each object
last_seen = {} #store the last seen frame of each object (key = track_id, value = last seen frame)
last_position = {} #store the last position of each object (key = track_id, value = (x, y) coordinates)
speed_history = {} #store the speed history of each object (key = track_id, value = recent speed measurements)
kalman_filters = {} #store the Kalman filter for each object (key = track_id, value = filter for that object)
predictions = {} #store the predictions for each object (key = track_id, value = dictionary of predictions for that object - key = prediction_horizon, value = (x, y) coordinates))

current_frame = 0
max_missing_frames = 30 #max frames an object can be missing before it is considered lost
prediction_horizons = [0.5, 1.0, 2.0] #seconds into the future to predict the position of the object

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
                kalman_filters[track_id] = create_kalman_filter( #initialize the Kalman filter for this object
                    current_x,
                    current_y,
                    dt
                )
                predictions[track_id] = {} #initialize the predictions for this object

            kf = kalman_filters[track_id] #get the Kalman filter for this object
            prediction = kf.predict() #predict where the object is now

            measurement = np.array([[np.float32(current_x)], [np.float32(current_y)]]) #create a measurement vector from the current position
            corrected = kf.correct(measurement) #correct the prediction with the measurement

            #store the filtered position (after Kalman filter correction) for this object
            filtered_x = int(corrected[0][0])
            filtered_y = int(corrected[1][0])

            filtered_vx = float(corrected[2][0])
            filtered_vy = float(corrected[3][0])

            predictions[track_id] = {} #reset the predictions for this object - every frame we want to recalculate the predictions based on the new filtered position and velocity  n 

            #predict the future position of the object in this framefor each prediction horizon
            for horizon in prediction_horizons:
                predicted_x, predicted_y = predict_future_position(
                    filtered_x,
                    filtered_y,
                    filtered_vx,
                    filtered_vy,
                    horizon
                )

                predictions[track_id][horizon] = (
                    predicted_x,
                    predicted_y
                )

            estimated_speed = np.sqrt(filtered_vx**2 + filtered_vy**2) #calculate the estimated speed from the filtered velocity components

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
                        "filtered_x": filtered_x,
                        "filtered_y": filtered_y,
                        "filtered_vx_px_per_sec": filtered_vx,
                        "filtered_vy_px_per_sec": filtered_vy,
                        "estimated_speed_px_per_sec": estimated_speed,
                        "dx": dx,
                        "dy": dy,
                        "distance_px": distance,
                        "frame_difference": frame_difference,
                        "time_difference_sec": time_difference,
                        "speed_px_per_sec": speed_px_per_sec,
                        "smoothed_speed_px_per_sec": smoothed_speed,
                        "filtered_x": filtered_x,
                        "filtered_y": filtered_y,
                        "predictions": predictions[track_id]
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
        kalman_filters.pop(track_id, None)
        predictions.pop(track_id, None)

    annotated_frame = result.plot()
    
    
    #draw historical trajectories
    for track_id, points in track_history.items():
        if len(points) > 1:
            points_array = np.array(
                points,
                dtype=np.int32
            )
    
            cv.polylines(
                annotated_frame,
                [points_array],
                isClosed=False,
                color=(255, 0, 0),
                thickness=5
            )
    
    #draw current filtered state and future predictions
    for track_id, data in telemetry.items():
        #skip tracks that do not have a filtered state yet
        if "filtered_x" not in data:
            continue
        
        #draw current velocity vector
        velocity_scale = 0.5
    
        end_x = int(
            data["filtered_x"]
            + data["filtered_vx_px_per_sec"] * velocity_scale
        )
    
        end_y = int(
            data["filtered_y"]
            + data["filtered_vy_px_per_sec"] * velocity_scale
        )
    
        cv.arrowedLine(
            annotated_frame,
            (
                data["filtered_x"],
                data["filtered_y"]
            ),
            (end_x, end_y),
            (0, 255, 0),
            3
        )
    
        #draw current filtered position
        cv.circle(
            annotated_frame,
            (
                data["filtered_x"],
                data["filtered_y"]
            ),
            radius=9,
            color=(0, 0, 255),
            thickness=-1
        )
    
        #skip future prediction drawing if predictions are unavailable
        if "predictions" not in data:
            continue
        
        #start future trajectory at current filtered position
        prediction_points = [
            (
                data["filtered_x"],
                data["filtered_y"]
            )
        ]
    
        #add each future prediction
        for horizon in prediction_horizons:
            predicted_x, predicted_y = data["predictions"][horizon]
            prediction_points.append(
                (predicted_x, predicted_y)
            )
    
            #draw predicted position
            cv.circle(
                annotated_frame,
                (predicted_x, predicted_y),
                radius=5,
                color=(0, 255, 255),
                thickness=-1
            )
            #label prediction time
            cv.putText(
                annotated_frame,
                f"+{horizon}s",
                (predicted_x + 5, predicted_y - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
    
        #convert prediction points to NumPy
        prediction_array = np.array(
            prediction_points,
            dtype=np.int32
        )
    
        #draw predicted trajectory
        cv.polylines(
            annotated_frame,
            [prediction_array],
            isClosed=False,
            color=(0, 255, 255),
            thickness=2
        )

    display_frame = cv.resize(annotated_frame, (1280, 720))
    cv.imshow("UAV Tracker", display_frame) #display the frame in a window

    if cv.waitKey(1) & 0xFF == ord('q'): #play the video (1ms frame switch) until 'q' is pressed
        break       

cap.release()
cv.destroyAllWindows()