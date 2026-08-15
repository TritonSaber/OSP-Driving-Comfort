from collections import defaultdict

import cv2 # OpenCV
import numpy as np
import time
import supervision as sv
import torch

from ultralytics import YOLO

def main():
    # Input video title (within path)
    print("Input driving video name (without .mp4): ")
    
    """
    # video_name = input()
    # video_path = "videos/" + video_name + ".mp4"
    """
    # TEMP: Just to make things quick since there is only one video
    video_path = "videos/makati_city_hall.mp4"


    print("Playing: " + video_path)

    # Best model training run weights
    model = YOLO("runs/detect/train/weights/best.pt")

    """
    # model = YOLO("yolo11m.pt")
    """

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
    feature_params = dict(maxCorners = 150,
                            qualityLevel = 0.03,
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
    if 'car_status_memory' not in locals(): car_status_memory = {} # save car status, whether stopped, moving, or neither
    scale_factor = 25

    while cap.isOpened():
        car_vectors_map = defaultdict(list) # reset for each frame, to store motion vectors for each car ID
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

        """
        # if detections.xyxy.shape[0] > 0:
        #     for tracker_id in detections.tracker_id:
        #         car_vectors_map[tracker_id] = []
        """
        
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
                
                src_pts = []
                dst_pts = []

                view_height, view_width = frame_gray.shape

                for i, (new, old) in enumerate(zip(good_next, good_prev)):
                    # new from good_next are points from new frame
                    # old from good_prev are points from old frame
                    # Both are used to check for the motion vectors, difference between old and new frames
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    motion_vector = new - old

                    # /// VISUAL: Draw a larger box so you see what is being tracked
                    # We are now tracking the "Middle Band" of the image
                    cv2.rectangle(frame, (0, int(view_height*0.15)), (view_width, int(view_height*0.9)), (0, 255, 0), 1)

                    if b < (view_height * 0.15) or b > (view_height * 0.90): continue
                    
                    # Filter any glitches within magnitude
                    if np.linalg.norm(motion_vector) > 100: continue 

                    # A boolean to check if car is inside the 
                    point_on_car = False
                    
                    if detections.xyxy.shape[0] > 0:
                        for j, box in enumerate(detections.xyxy):
                            x1, y1, x2, y2 = box
                            raw_id = detections.tracker_id[j]
                            current_id = int(raw_id)

                            if current_id not in car_vectors_map:
                                car_vectors_map[current_id] = []
                            
                            # If there is a point inside the bounding box, apply indication that the point inside the car using car_vectors_map
                            if x1 < a < x2 and y1 < b < y2:
                                """
                                # DEBUG: Print one example to check the scale
                                # if i == 0 and len(detections) > 0:
                                #     print(f"--- DEBUG COORDINATES ---")
                                #     print(f"Point: ({a}, {b})")
                                #     print(f"Box:   ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                                #     print(f"Is Inside? {x1 < a < x2 and y1 < b < y2}")
                                #     print(f"-------------------------")
                                """

                                
                                """
                                # car_vectors_map append
                                # if tracker_id not in car_vectors_map:
                                #     car_vectors_map[tracker_id] = []
                                """
                                car_vectors_map[current_id].append(motion_vector)
                                point_on_car = True
                                # mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 1)
                                # Visual: Draw RED dot for car points
                                cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)
                                # if i == 0 and j == 0: # Only print for the first point and first box to save console space
                                #     print(f"DEBUG CHECK:")
                                #     print(f"  Point: {a}, {b} (Type: {type(a)})")
                                #     print(f"  Box: {x1}, {y1}, {x2}, {y2} (Type: {type(x1)})")
                                break
                    
                    # If the point is NOT on a car, it is a regular point placed in the background
                    if not point_on_car:
                        vector = new - old
                        if vector[1] > 0:
                            background_vectors.append(motion_vector)
                        # mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 1) # just to show lines for the point
                        cv2.circle(frame, (a, b), 3, (0, 255, 0), -1)

                    dx = a - c
                    dy = b - d
                    # if dy < -0.5:
                    #     continue

                    if abs(dx) > 50 or abs(dy) > 50:
                        continue

                    background_vectors.append([dx, dy])

                # RANSAC (Random Sample Consensus)
                if len(src_pts) > 10:
                    src_pts = np.float32(src_pts)
                    dst_pts = np.float32(dst_pts)

                    transform_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

                    if len(src_pts) > 0 and transform_matrix is not None:
                        src_pts = src_pts[:200]
                        dst_pts = dst_pts[:200]

                    if transform_matrix is not None:
                        current_global_vector = transform_matrix[:, 2] # The translation component of the affine transformation
                    else:
                        current_global_vector = np.array([0., 0.])
                else:
                    current_global_vector = np.array([0., 0.])

                # IMPORTANT TO NOTE: this is for checking background vectors, and using it for the global motion vector (the background is the focus)
                if len(background_vectors) > 0:
                    current_global_vector = np.median(background_vectors, axis=0)
                else:
                    current_global_vector = np.array([0., 0.])
                    
                if 'smooth_global' not in locals():
                    smooth_global = current_global_vector
                else:
                    smooth_global = (smooth_global * 0.9) + (current_global_vector * 0.1)
                
                global_motion_vector = smooth_global

                # /// DEBUG: Print the entire map for one frame
                # Only print if we actually have cars detected to avoid spamming empty dicts
                # if len(detections) > 0:
                #     print(f"--- FRAME {frame_count} MAP CHECK ---")
                #     print(f"Detected IDs: {detections.tracker_id}")
                #     print(f"Map Keys:     {list(car_vectors_map.keys())}")
                    
                #     # Print how many vectors are inside each key
                #     for tid in car_vectors_map:
                #         count = len(car_vectors_map[tid])
                #         print(f"ID {tid}: {count} vectors")
                #     print("-----------------------------------")

                # Draw the Arrow
                # center = (int(frame.shape[1]*0.5), int(frame.shape[0]*0.1))
                # end_point = (int(center[0] + global_motion_vector[0]*7), int(center[1] + global_motion_vector[1]*7))
                # cv2.arrowedLine(frame, center, d, (0, 0, 255), 3, tipLength=0.5)
                
            else:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                mask = np.zeros_like(prev_frame)
                global_motion_vector = np.array([0, 0])

        # Global motion
        h, w = frame.shape[:2]
        cv2.arrowedLine(frame, (w//2, h//2), 
                       (int(w/2 + global_motion_vector[0]*scale_factor), int(h/2 + global_motion_vector[1]*5)), 
                       (0, 0, 255), 3, tipLength=0.3)

        status_labels = [] # the final label for the bounding box (using label annotator)

        # Proceed only if there are detections left after filtering
        if detections.xyxy.shape[0] > 0:
            for i in range(detections.xyxy.shape[0]):
                raw_id = detections.tracker_id[i]
                current_id = int(raw_id)
                bbox = detections.xyxy[i]

                my_vectors = car_vectors_map.get(current_id, []) # my_vectors - any vector within a bounding box
                
                if len(my_vectors) < 3:
                    current_score = car_consistency.get(current_id, 0)

                    if current_score > 4:
                        status = "ACTIVE"
                    else:
                        status = "STOPPED"
                else:
                    # avg_car_vector via the median
                    avg_car_vector = np.median(my_vectors, axis=0)
                    
                    # start of optical flow comparison
                    
                    # for testing
                    diff_vector = avg_car_vector - global_motion_vector
                    
                    diff_vector_x = avg_car_vector[0] - global_motion_vector[0]
                    diff_vector_y = avg_car_vector[1] - global_motion_vector[1]

                    global_speed_y = abs(global_motion_vector[1])
                    magnitude_diff = np.linalg.norm(diff_vector)
                    temp_class_id = detections.class_id[i]
                    temp_class_name = model.names[temp_class_id]
                    
                    # dynamic threshold - allowing more leeway if 
                    global_speed = np.linalg.norm(global_motion_vector)
                    threshold = 0.8 + (global_speed * 0.2)

                    # Y-threshold and X-threshold
                    # X-threshold is heavier since x is less important (might change)
                    threshold_y = 0.6 + (global_speed * 0.2)
                    threshold_x = 0.8 + (global_speed * 0.5)

                    is_moving_y = abs(diff_vector_y) > threshold_y
                    is_moving_x = abs(diff_vector_x) > threshold_x

                    is_moving = False

                    # Consistency Logic
                    if current_id not in car_consistency: car_consistency[current_id] = 0
                    
                    # if magnitude_diff > threshold:
                    #     car_consistency[current_id] += 1
                    # else:
                    #     car_consistency[current_id] -= 1

                    if abs(is_moving_y) > abs(is_moving_x) and abs(is_moving_y) > threshold:
                        is_moving = True
                    elif abs(is_moving_x) > abs(is_moving_y) and abs(is_moving_x) > (threshold * 1.2):
                        is_moving = True

                    if is_moving:
                        car_consistency[current_id] += 1
                    else:
                        car_consistency[current_id] -= 1
                        
                    car_consistency[current_id] = max(-5, min(10, car_consistency[current_id]))
                    previous_status = car_status_memory.get(current_id, "UNCERTAIN")

                    # Hysteresis check - checks for status of vehicle, in between 0 and 4
                    # would allow for box to not immediately bounce from one status to another too quickly
                    # previous_status would be the fallback if that happens
                    if car_consistency[current_id] >= 4:
                        status = "ACTIVE"
                    elif car_consistency[current_id] <= 0:
                        status = "STOPPED"
                    else:
                        status = previous_status
                    
                    # update car status
                    car_status_memory[current_id] = status

                    # DEBUG FOR CHECKING car_consistency 
                    mov_track = car_consistency[current_id]

                    # VISUAL DEBUG: Draw the Car Arrow (Blue)
                    cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
                    cv2.arrowedLine(frame, (cx, cy),
                                   (int(cx + avg_car_vector[0]*scale_factor), int(cy + avg_car_vector[1]*5)), 
                                   (255, 255, 0), 2, tipLength=0.3)

                    """
                    # Print Diff Score
                    # cv2.putText(frame, f"{magnitude_diff:.1f}/{threshold:.1f}", (int(bbox[0]), int(bbox[1])-5), 
                    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # debug_target_index = (frame_count // 120) % len(detections)
                    # if i == debug_target_index: 
                    #     print(f"--- ID {current_id} CN {temp_class_name} ---")
                    #     print(f"Global Vector: {global_motion_vector} (Length: {global_speed:.2f})")
                    #     print(f"Car Vector:    {avg_car_vector} (Length: {np.linalg.norm(avg_car_vector):.2f})")
                    #     print(f"Difference:    {magnitude_diff:.2f}")
                    #     print(f"Threshold:     {0.5 + (np.linalg.norm(global_motion_vector) * 0.2):.2f}")
                    #     print(f"-------------------")
                        # print(f"--- ID {current_id} ---")
                        # print(f"Diff Y: {diff_vector_y:.2f} (Thresh: {threshold_y:.2f}) -> Moving? {is_moving_y}")
                        # print(f"Diff X: {diff_vector_x:.2f} (Thresh: {threshold_x:.2f}) -> Moving? {is_moving_x}")
                    """
                        
            for i in range(detections.xyxy.shape[0]):
                conf = detections.confidence[i]
                class_id = detections.class_id[i]
                class_name = model.names[class_id]
                detect_id = detections.tracker_id[i]

                print(f"ID: {detect_id} | Points: {len(my_vectors)} | Diff: {magnitude_diff:.2f} | Thresh: {threshold:.2f}")
                
                # final label - class name, confidence for box, status (STOPPED, ACTIVE, etc.), car consistency (TEMPORARY)
                final_label = f"{detect_id} {class_name} {conf:.2f} {status} {mov_track}"
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
