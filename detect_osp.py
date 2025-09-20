import cv2
import numpy as np
import time
import supervision as sv

from ultralytics import YOLO

model = YOLO("yolov11m.pt")
cap = cv2.VideoCapture("your_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 4/fps

# class initialize_osp(model):
#     def initializeYOLO():
#         # test

'''
class calculate_osp():
    def calculateVelocity():
        while cap.isOpened():
            trackers = {}

            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, tracker="botsort.yaml")

            if results and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy()
                positions = results[0].boxes.xywh.cpu.numpy()

                for obj_id, pos in zip(ids, positions):
                    center_x, center_y = pos[0], pos[1]
                    current_time = time.time()

                    if obj_id in trackers:
                        prev_pos, prev_time = trackers[obj_id]
                        dx = center_x - prev_pos[0]
                        dy = center_y - prev_pos[1]
                        pixel_distance = np.sqrt(dx**2 + dy**2)
                        elapsed_time = current_time - prev_time
                        velocity = pixel_distance / elapsed_time

                trackers[obj_id] = ((center_x,center_y), current_time)


                cv2.imshow('Parking Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
'''

def main():
    # Input video title (within path)
    print("Input driving video name (without .mp4): ")
    video_name = input()
    video_path = "video/" + video_name + ".mp4"

    # Best model weight run
    model_path = "runs/detect/train/weights/best.pt"

    model = YOLO(model_path)

    tracker = sv.ByteTrack()

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        result = model(frame, classes=[2], verbose=False)[0]

        # change results to 
        detections = sv.Detections.from_ultralytics(result)

        # confidence of detections
        detections = tracker.update_with_detections(detections)

        labels = [
            f"ID:{tracker_id}"
            for tracker_id
            in detections
        ]
        
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
            labels=labels
        )

        cv2.imshow("On-Street Parking Detection", annotated_frame)

        # Press 'q' to quit running the program by breaking the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
