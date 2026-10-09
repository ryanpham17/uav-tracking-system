import numpy as np
import cv2 as cv
from ultralytics import YOLO as yolo

def create_kalman_filter(x, y, dt):
    kf = cv.KalmanFilter(4, 2)

    # The state is [x, y, vx, vy], where position is measured in pixels and
    # velocity is measured in pixels per second.
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # The detector provides position only; velocity is estimated by the filter.
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    # Lower process noise favors smooth, constant-velocity motion.
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    # Measurement noise controls how strongly noisy detector positions are trusted.
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

    # Start at the first detected position with no initial velocity estimate.
    kf.statePost = np.array([
        [x],
        [y],
        [0],
        [0]
    ], dtype=np.float32)

    return kf

# Predict a future pixel position from the filtered position and velocity.
def predict_future_position(x, y, vx, vy, time_ahead):
    future_x = x + vx * time_ahead
    future_y = y + vy * time_ahead

    return int(future_x), int(future_y)

# Compare predictions that have reached their target frame with the observed position.
def evaluate_prediction(
        track_id,
        current_frame,
        current_x,
        current_y,
        prediction_history,
        prediction_results
):
    remaining_predictions = []

    for prediction in prediction_history[track_id]:
        if current_frame >= prediction["target_frame"]:
            error_x = (
                prediction["predicted_x"] - current_x
            )
            error_y = (
                prediction["predicted_y"] - current_y
            )
            
            error = np.sqrt(error_x**2 + error_y**2)

            prediction_results[track_id].append({
                "horizon": prediction["horizon"],
                "error_px": error
            })
        else:
            remaining_predictions.append(prediction)

def calculate_prediction_metrics(prediction_results):
    horizon_errors = {}

    # TODO: Aggregate completed errors by prediction horizon before using this helper.
    prediction_history[track_id] = remaining_predictions

# Select the model used for this run. The bundled nano model is configured for
# testing with cars; replace it with the drone model for UAV detection.
# model = yolo("models/drone-yolo26m.pt")
model = yolo("models/yolo26n.pt")

# Open the input video. A camera index such as 0 can be used instead of a file path.
cap = cv.VideoCapture("videos/testroad.mp4") 

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv.CAP_PROP_FPS)
dt = 1 / fps  # Time between frames, used by the motion model.

# Per-track state. Every dictionary is keyed by the detector's persistent track ID.
track_history = {}       # Recent observed center points, used to draw trajectories.
telemetry = {}           # Latest motion measurements exposed for each track.
last_seen = {}           # Frame number in which each track was most recently detected.
last_position = {}       # Most recent raw center point and frame number.
speed_history = {}       # Recent raw speed samples used for smoothing.
kalman_filters = {}      # Kalman filter that estimates position and velocity.
predictions = {}         # Latest future positions, keyed by prediction horizon in seconds.
prediction_history = {}  # Predictions waiting to be evaluated at their target frame.
prediction_results = {}  # Position error recorded for completed predictions.

current_frame = 0
max_missing_frames = 30  # Remove a track after this many consecutive missing frames.
prediction_horizons = [0.5, 1.0, 2.0]  # Future prediction horizons, in seconds.

# Process the input one frame at a time until the stream ends or the user quits.
while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Could not read frame (stream ended). Exiting...")
        break

    current_frame += 1

    # Detect objects and update ByteTrack's persistent IDs for this frame.
    result = model.track(
        frame,
        persist = True,
        tracker = "bytetrack.yaml",
        imgsz = 704,
        conf = 0.20,
        verbose = False,
        device = 0,
        quantize = 16)[0] 

    # Update state for each object that was detected in the current frame.
    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xywh.cpu().tolist()  # [center_x, center_y, width, height]

        # Initialize all per-track containers the first time an ID appears.
        for box, track_id in zip(boxes, track_ids):
            x, y, width, height = box
            current_x = int(x)
            current_y = int(y)
            last_seen[track_id] = current_frame

            if track_id not in track_history:
                track_history[track_id] = []
                telemetry[track_id] = {}
                speed_history[track_id] = []
                kalman_filters[track_id] = create_kalman_filter(
                    current_x,
                    current_y,
                    dt
                )
                predictions[track_id] = {}
                prediction_results[track_id] = []
                prediction_history[track_id] = []

            evaluate_prediction(
                track_id,
                current_frame,
                current_x,
                current_y,
                prediction_history,
                prediction_results
            )

            kf = kalman_filters[track_id]
            prediction = kf.predict()  # Estimate the current state before applying the new measurement.

            measurement = np.array([[np.float32(current_x)], [np.float32(current_y)]])
            corrected = kf.correct(measurement)

            # Read the smoothed position and velocity after the measurement update.
            filtered_x = int(corrected[0][0])
            filtered_y = int(corrected[1][0])

            filtered_vx = float(corrected[2][0])
            filtered_vy = float(corrected[3][0])

            # Rebuild predictions every frame from the latest filtered state.
            predictions[track_id] = {}

            # Generate a prediction and evaluation record for each configured horizon.
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

                target_frame = current_frame + int(round(horizon * fps))

                prediction_history[track_id].append({
                    "created_frame": current_frame,
                    "horizon": horizon,
                    "target_frame": target_frame,
                    "predicted_x": predicted_x,
                    "predicted_y": predicted_y
                })

            estimated_speed = np.sqrt(filtered_vx**2 + filtered_vy**2)

            track_history[track_id].append((current_x, current_y))

            # Calculate motion relative to the previous observation of this track.
            if track_id in last_position: 
                previous_x, previous_y, previous_frame = last_position[track_id]

                dx = current_x - previous_x
                dy = current_y - previous_y

                distance = np.sqrt(dx**2 + dy**2)

                frame_difference = current_frame - previous_frame

                if frame_difference > 0:
                    time_difference = frame_difference / fps
                    speed_px_per_sec = distance / time_difference
                    speed_history[track_id].append(speed_px_per_sec)

                    if len(speed_history[track_id]) > 5:
                        speed_history[track_id].pop(0)

                    smoothed_speed = np.mean(speed_history[track_id])

                    # Store raw, filtered, and predicted motion data for downstream use.
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
                        "predictions": predictions[track_id]
                    }

            last_position[track_id] = (current_x, current_y, current_frame)
                
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

    # Remove tracks that have not been observed recently.
    stale_tracks = []

    for track_id, last_frame in last_seen.items():
        if current_frame - last_frame > max_missing_frames:
            stale_tracks.append(track_id)

    # Keep all per-track dictionaries synchronized when a track expires.
    for track_id in stale_tracks:
        track_history.pop(track_id, None)
        telemetry.pop(track_id, None)
        last_seen.pop(track_id, None)
        last_position.pop(track_id, None)
        speed_history.pop(track_id, None)
        kalman_filters.pop(track_id, None)
        predictions.pop(track_id, None)
        prediction_history.pop(track_id, None)

    annotated_frame = result.plot()
    
    # Draw the recent observed trajectory for each active track.
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
    
    # Draw each track's filtered state, velocity vector, and future predictions.
    for track_id, data in telemetry.items():
        # Telemetry is populated only after a previous observation exists.
        if "filtered_x" not in data:
            continue
        
        # Scale the velocity vector so it remains visible at video resolution.
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
    
        # Mark the current filtered position.
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
    
        # Some tracks may not yet have telemetry with prediction data.
        if "predictions" not in data:
            continue
        
        # Start the predicted path at the current filtered position.
        prediction_points = [
            (
                data["filtered_x"],
                data["filtered_y"]
            )
        ]
    
        # Add and label each configured future position.
        for horizon in prediction_horizons:
            predicted_x, predicted_y = data["predictions"][horizon]
            prediction_points.append(
                (predicted_x, predicted_y)
            )
    
            # Mark the predicted position.
            cv.circle(
                annotated_frame,
                (predicted_x, predicted_y),
                radius=5,
                color=(0, 255, 255),
                thickness=-1
            )
            # Label the prediction horizon in seconds.
            cv.putText(
                annotated_frame,
                f"+{horizon}s",
                (predicted_x + 5, predicted_y - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
    
        # Convert the points before passing them to OpenCV.
        prediction_array = np.array(
            prediction_points,
            dtype=np.int32
        )
    
        # Draw the projected trajectory.
        cv.polylines(
            annotated_frame,
            [prediction_array],
            isClosed=False,
            color=(0, 255, 255),
            thickness=2
        )

    display_frame = cv.resize(annotated_frame, (1280, 720))
    cv.imshow("UAV Tracker", display_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):  # Press Q to stop playback.
        break       

cap.release()
cv.destroyAllWindows()