import ultralytics
import torch
from roboflow import Roboflow

from ultralytics import YOLO

def main():
    # This checks if ultralytics (and YOLO) is working
    ultralytics.checks()

    # This checks if CUDA is available
    torch.cuda.is_available()

    # The Driving Comfort dataset, version 1 of the dataset with no pedestrian label
    rf = Roboflow(api_key="IxXkXChvKvzhRnqjmm23")
    project = rf.workspace("my-workspace-wvghr").project("-no-peds-driving-comfort-detection")
    version = project.version(1)
    dataset = version.download("yolov11")

    model = YOLO('yolo11s.pt')

    print(model.names)

    
    train_results = model.train(
        data=f'{dataset.location}/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8, # batch size reduced from 16 to 8 for VRAM
        workers=2, # workers reduced from 16 to 4 for RAM
        device=0
    )

    '''
        NOTES:
        1. YOLO11n - training time of 0.360 hours (run 1), 0.287 hours (run 2)
        2. YOLO11s - training time of 0.331 hours (run 1)
        3. YOLO11m - training time of 0.753 hours (run 1), 1.773 hours (run 2), 0.760 hours (run 3)
        4. YOLO11l - training time of 2.590 hours (run 1), 2.031 hours (run 2)
    '''

if __name__ == "__main__":
    main()