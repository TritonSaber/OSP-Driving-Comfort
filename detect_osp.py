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

    tracker = sv.ByteTrack()

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

        # Removes frames without tracker ID
        if detections.tracker_id.any():
            mask = [tracker_id is not None for tracker_id in detections.tracker_id]
            detections = detections[(detections.confidence > 0.5) & mask]
        
        # Proceed only if there are detections left after filtering
        if detections.xyxy.shape[0] > 0:
            labels = [
                f"TID:{tracker_id} CID:{class_id} {class_name} {confidence:0.2f}"
                for tracker_id, class_id, class_name, confidence
                in zip(detections.tracker_id, detections.class_id, detections['class_name'], detections.confidence)
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
