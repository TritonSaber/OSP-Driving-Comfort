import cv2 # OpenCV
import numpy as np
import time
import supervision as sv

from ultralytics import YOLO

def main():
    # Input video title (within path)
    print("Input driving video name (without .mp4): ")
    # video_name = input()
    # video_path = "videos/" + video_name + ".mp4"

    # TEMP: Just to make things quick since there is only one video
    video_path = "videos/makati_city_hall.mp4"


    print("Playing: " + video_path)

    # Best model training run weights
    model = YOLO("runs/detect/train/weights/best.pt")

    video_info = sv.VideoInfo.from_video_path(video_path)
    tracker = sv.ByteTrack(frame_rate=video_info.fps)
    smoother = sv.DetectionsSmoother()

    # Bounding box annotation
    box_annotator = sv.BoxAnnotator(
        thickness=2
    )

    # Label annotation
    label_annotator=sv.LabelAnnotator()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Optical Flow setup, using Lucas-Kanade Optical Flow, meaning this is sparse, not dense
    # ShiTomasi corner detection parameters
    feature_params = dict(maxCorners = 200,
                            qualityLevel = 0.1,
                            minDistance = 7,
                            blockSize = 7)
    
    # Lucas-Kanade Optical Flow parameters
    lk_params = dict(winSize = (15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # To detect corners, get current frame before using next 30 frames for creating new points connected by a line
    ret, prev_frame = cap.read()
    if not ret:
        # If there are no frames in the video
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # cv2.goodFeaturesToTrack - finds N strongest corners in the image
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

    # mask image to draw lines
    mask = np.zeros_like(prev_frame)
    frame_count = 0

    car_consistency = {}
    car_vectors_map = {} # temp bucket for optical flow arrows by Car ID for current frame
    car_vectors_smoother = {}
    if 'car_status_memory' not in locals(): car_status_memory = {} # save car status, whether stopped, moving, or neither

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # If there are no more frames (video has ended)
            break
        
        result = model(frame, verbose=False)[0]

        # Load in predictions to Supervision, specifically Ultralytics because inference model comes from there
        detections = sv.Detections.from_ultralytics(result)

        # Load in detection result to ByteTrack
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        # Filter that removes frames without tracker ID
        if detections.tracker_id.any():
            mask_filter = [tracker_id is not None for tracker_id in detections.tracker_id]
            detections = detections[(detections.confidence > 0.3) & mask_filter]

        global_motion_vector = np.array([0, 0])

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        background_vectors = []

        if detections.xyxy.shape[0] > 0:
            for tracker_id in detections.tracker_id:
                car_vectors_map[tracker_id] = []

        # check to ignore if there are no p0 points
        if p0 is not None:
            # cv2.calcOpticalFlowPyrLK - for optical flow between two frames
            # p1 - new positions of p0 points, st - status of flow between points, err - errors
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None and st is not None:
                good_next = p1[st==1] # good_next - good points from next frame, st == 1 means flow between points exists
                good_prev = p0[st==1] # good_prev - good points from previous frame

                """
                clean_motion_vectors = []

                # GET FRAME DIMENSIONS FOR EDGE FILTERING
                h, w = frame.shape[:2]
                edge_margin = 20 # Ignore points within 20px of edge
                
                # SPEED LIMIT (Magnitude Cap)
                # If a point moves more than 20 pixels in 1 frame, it's likely a glitch/error
                max_jump_limit = 20.0
                """

                for i, (new, old) in enumerate(zip(good_next, good_prev)):
                    # (a, b) are points from new frame, (c, d) are points from old frame, used for line
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    motion_vector = new - old
                    
                    # Filter any glitches within magnitude
                    if np.linalg.norm(motion_vector) > 100: continue 

                    # CHECK: Is this point inside a Car Box?
                    point_on_car = False
                    
                    if detections.xyxy.shape[0] > 0:
                        for j, box in enumerate(detections.xyxy):
                            x1, y1, x2, y2 = box
                            tracker_id = detections.tracker_id[j]
                            
                            # If there is a point inside the bounding box, apply indication that the point inside the car using car_vectors_map
                            if x1 < a < x2 and y1 < b < y2:
                                # DEBUG: Print one example to check the scale
                                # if i == 0 and len(detections) > 0:
                                #     print(f"--- DEBUG COORDINATES ---")
                                #     print(f"Point: ({a}, {b})")
                                #     print(f"Box:   ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                                #     print(f"Is Inside? {x1 < a < x2 and y1 < b < y2}")
                                #     print(f"-------------------------")
                                
                                # car_vectors_map append
                                car_vectors_map[tracker_id].append(motion_vector)
                                point_on_car = True
                                # mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 1)
                                # Visual: Draw RED dot for car points
                                cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)
                                # if i == 0 and j == 0: # Only print for the first point and first box to save console space
                                #     print(f"DEBUG CHECK:")
                                #     print(f"  Point: {a}, {b} (Type: {type(a)})")
                                #     print(f"  Box: {x1}, {y1}, {x2}, {y2} (Type: {type(x1)})")
                                break
                    
                    # If the point is NOT on a car, it is a regular point placed in the background
                    if not point_on_car:
                        background_vectors.append(motion_vector)
                        # mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 1) # just to show lines for the point
                        cv2.circle(frame, (a, b), 3, (0, 255, 0), -1)

                # Draw the Arrow
                # center = (int(frame.shape[1]*0.5), int(frame.shape[0]*0.1))
                # end_point = (int(center[0] + global_motion_vector[0]*7), int(center[1] + global_motion_vector[1]*7))
                # cv2.arrowedLine(frame, center, d, (0, 0, 255), 3, tipLength=0.5)
                
            else:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                mask = np.zeros_like(prev_frame)
                global_motion_vector = np.array([0, 0])

        global_motion_vector = np.array([0.0, 0.0])
        # IMPORTANT TO NOTE:
        if len(background_vectors) > 0:
            global_motion_vector = np.median(background_vectors, axis=0)

        # Global motion
        h, w = frame.shape[:2]
        cv2.arrowedLine(frame, (w//2, h//2), 
                       (int(w/2 + global_motion_vector[0]*5), int(h/2 + global_motion_vector[1]*5)), 
                       (0, 0, 255), 3, tipLength=0.3)

        status_labels = [] # the final label for the bounding box (using label annotator)

        # Proceed only if there are detections left after filtering
        if detections.xyxy.shape[0] > 0:
            for i in range(detections.xyxy.shape[0]):
                tracker_id = detections.tracker_id[i]
                bbox = detections.xyxy[i]

                my_vectors = car_vectors_map.get(tracker_id, []) # my_vectors - any vector within a bounding box

                if len(my_vectors) < 3:
                    # Vehicle cannot be detected properly, do not put any status
                    status = "NONE"
                    avg_car_vector = np.array([0.0, 0.0])
                else:
                    # avg_car_vector via the median
                    avg_car_vector = np.median(my_vectors, axis=0)
                    
                    # start of optical flow comparison
                    diff_vector = avg_car_vector - global_motion_vector
                    magnitude_diff = np.linalg.norm(diff_vector)
                    
                    # dynamic threshold - allowing more leeway if 
                    global_speed = np.linalg.norm(global_motion_vector)
                    threshold = 1.5 + (global_speed * 0.5)
                    
                    # Consistency Logic
                    if tracker_id not in car_consistency: car_consistency[tracker_id] = 0
                    
                    if magnitude_diff > threshold:
                        car_consistency[tracker_id] += 1
                    else:
                        car_consistency[tracker_id] -= 1
                        
                    car_consistency[tracker_id] = max(-5, min(10, car_consistency[tracker_id]))
                    previous_status = car_status_memory.get(tracker_id, "NONE")

                    # Hysteresis check - checks for status of vehicle, in between 0 and 4
                    # would allow for box to not immediately bounce from one status to another too quickly
                    # previous_status would be the fallback if that happens
                    if car_consistency[tracker_id] >= 4:
                        status = "ACTIVE"
                    elif car_consistency[tracker_id] <= 0:
                        status = "STOPPED"
                    else:
                        status = previous_status
                    
                    # update car status
                    car_status_memory[tracker_id] = status

                    # DEBUG FOR CHECKING car_consistency 
                    mov_track = car_consistency[tracker_id]

                    # VISUAL DEBUG: Draw the Car Arrow (Blue)
                    cx, cy = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
                    cv2.arrowedLine(frame, (cx, cy),
                                   (int(cx + avg_car_vector[0]*5), int(cy + avg_car_vector[1]*5)), 
                                   (255, 255, 0), 2, tipLength=0.3)
                    
                    # Print Diff Score
                    cv2.putText(frame, f"{magnitude_diff:.1f}/{threshold:.1f}", (int(bbox[0]), int(bbox[1])-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            for i in range(detections.xyxy.shape[0]):
                conf = detections.confidence[i]
                class_id = detections.class_id[i]
                class_name = model.names[class_id]
                
                # final label - class name, confidence for box, status (STOPPED, ACTIVE, etc.), car consistency (TEMPORARY)
                final_label = f"{class_name} {conf:.2f} {status} {mov_track}"
                status_labels.append(final_label)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
                # color=color
            )

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=status_labels
                # color=color
            )

            # Show annotated frame using bounding boxes and labels
            annotated_frame = cv2.add(annotated_frame, mask)
            cv2.imshow("On-Street Parking Detection", annotated_frame)

        else:
            # Otherwise, show the original frame in a different window if there are no detections
            cv2.imshow("On-Street Parking Detection", frame)

        if p1 is not None:
            p0 = good_next.reshape(-1, 1, 2)
        else:
            p0 = None
        # if 'good_next' in locals() and good_next.shape[0] > 0 else cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

        frame_count += 1
        # Reset mask and older point to the second most recent frame
        if frame_count % 30 == 0:
            mask = np.zeros_like(prev_frame)
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        
        elif p0 is None or len(p0) < 100:
            mask = np.zeros_like(prev_frame)
            new_points = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

            if new_points is not None:
                if p0 is not None:
                    p0 = np.concatenate((p0, new_points), axis=0)
                else:
                    p0 = new_points


        prev_gray = frame_gray.copy()

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
