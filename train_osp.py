# import ultralytics
import torch
import os

from roboflow import Roboflow
from dotenv import load_dotenv
from pathlib import Path
from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema

from rfdetr import RFDETRMedium, RFDETRSmall, RFDETRNano

def main():
    # Load .env
    load_dotenv(".env")

    # This checks if ultralytics (and YOLO) is working
    # ultralytics.checks()

    # This checks if CUDA is available
    torch.cuda.is_available()

    # The Driving Comfort dataset, version 1 of the dataset with no pedestrian label
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE")).project(os.getenv("ROBOFLOW_PROJECT"))
    version = project.version(os.getenv("ROBOFLOW_VERSION"))
    dataset = version.download(os.getenv("ROBOFLOW_DATASET_VERSION"))

    DATASET_DIR = Path(dataset.location)
    # schema = infer_coco_keypoint_schema(DATASET_DIR / "train" / "_annotations.coco.json")

    # model = RFDETR

    # Mainly loading RFDETR Medium for training, alternatively can load nano or small for faster training
    # Originally YOLO11m was used for training, but RF-DETR has better performance
    model = RFDETRMedium()

    # print(model.names)

    # Change data to COCO if needed

    """

    ***SETTINGS:***
    - model: main model used is RFDETRMedium(), change to RFDETRNano() or RFDETRSmall() for faster training, but lower accuracy
    - resume: add "resume=runs/detect/[output folder]/last.ckpt" to resume training from last checkpoint
    - output_dir: change output directory to "runs/detect/[output folder]" depending on where you want to save the output
    - early_stopping: set to True to enable early stopping, set to False to disable early stopping, halts training if 
    validation loss does not improve for a number of epochs
    - early_stopping_patience: set to the number of epochs to wait for early stopping, default is 15

    """
    train_results = model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=100,
        batch_size=4,
        grad_accum_steps=1,
        lr=1e-4,
        output_dir="runs/detect/train_rfdetr", 
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