import cv2 # OpenCV
import numpy as np
import time
import supervision as sv

from ultralytics import YOLO

def main():
    # Input video title (within path)
    print("Input driving video name (without .mp4): ")
    video_name = input()
    video_path = "videos/" + video_name + ".mp4"

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
    feature_params = dict(maxCorners = 100,
                            qualityLevel = 0.3,
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
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # If there are no more frames (video has ended)
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.calcOpticalFlowPyrLK - for optical flow between two frames
        # p1 - new positions of p0 points, st - status of flow between points, err - errors
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None and st is not None:
            good_next = p1[st==1] # good_next - good points from next frame, st == 1 means flow between points exists
            good_prev = p0[st==1] # good_prev - good points from previous frame

            motion_vectors = good_next - good_prev # motion_vectors - distance between two similar points in two different frames

            # if there is distance between motion vectors from different frames
            if motion_vectors.shape[0] > 0:
                global_motion_vector = np.median(motion_vectors, axis=0) # main median vector denoting the motion of the camera (based on the vehicle)
                
                for i, (new, old) in enumerate(zip(good_next, good_prev)):
                    # (a, b) are points from new frame, (c, d) are points from old frame
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 1)
                    frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

                # Arrow denoting the movement of the vehicle based on the global motion vector
                center = (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.1))
                end_point = (int(center[0] + global_motion_vector[0] * 10), int(center[1] + global_motion_vector[1] * 10))
                frame = cv2.arrowedLine(frame, center, end_point, (0, 0, 255), 3, tipLength=0.5)

            # otherwise, if there are no points, reset global motion vector to 0, 0
            else:
                global_motion_vector = np.array([0, 0])

        else:
            # If there are no points, create new points to start with
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
            mask = np.zeros_like(prev_frame)
            global_motion_vector = np.array([0, 0])

        prev_gray = frame_gray.copy()
        p0 = good_next.reshape(-1, 1, 2) if 'good_next' in locals() and good_next.shape[0] > 0 else cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

        frame_count += 1
        # Reset mask and older point to the second most recent frame
        if frame_count % 30 == 0:
            mask = np.zeros_like(prev_frame)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

        result = model(frame, verbose=False)[0]

        # Load in predictions to Supervision, specifically Ultralytics because inference model comes from there
        detections = sv.Detections.from_ultralytics(result)

        # Load in detection result to ByteTrack
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        # Removes frames without tracker ID
        if detections.tracker_id.any():
            mask_filter = [tracker_id is not None for tracker_id in detections.tracker_id]
            detections = detections[(detections.confidence > 0.3) & mask_filter]
        
        # Proceed only if there are detections left after filtering
        if detections.xyxy.shape[0] > 0:
            labels = [
                f"{tracker_id} {class_name} {confidence:0.2f}"
                for tracker_id, class_name, confidence
                in zip(detections.tracker_id, detections['class_name'], detections.confidence)
            ]
            
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )

            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

             # Show annotated frame using bounding boxes and labels
            cv2.imshow("On-Street Parking Detection", annotated_frame)

        else:
            # Otherwise, show the original frame in a different window if there are no detections
            cv2.imshow("On-Street Parking Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
