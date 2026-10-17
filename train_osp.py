# import ultralytics
import torch
import os

from roboflow import Roboflow
from dotenv import load_dotenv
from pathlib import Path
from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema
from time import perf_counter

from rfdetr import RFDETRBase, RFDETRMedium, RFDETRSmall, RFDETRNano

def main():
    # Load .env
    load_dotenv(".env")

    # This checks if CUDA is available
    torch.cuda.is_available()

    # The Driving Comfort dataset, version 1 of the dataset with no pedestrian label
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE")).project(os.getenv("ROBOFLOW_PROJECT"))
    version = project.version(os.getenv("ROBOFLOW_VERSION"))
    dataset = version.download(os.getenv("ROBOFLOW_DATASET_VERSION"))

    DATASET_DIR = Path(dataset.location)
    # schema = infer_coco_keypoint_schema(DATASET_DIR / "train" / "_annotations.coco.json")
    
    # Mainly loading RFDETRMedium() for training, alternatively can load RFDETRNano() or RFDETRSmall() for faster training
    # Alternatively, RFDETRBase uses the DINOv2-compatible patch and positional-encoding configuration.
    # Originally YOLO11m was used for training, but RF-DETR has better performance at the cost of longer training time
    model = RFDETRMedium(num_classes=dataset.num_classes)

    start = perf_counter()

    """

    ***SETTINGS:***
    - model: main model used is RFDETRMedium(), change to RFDETRNano() or RFDETRSmall() for faster training, but lower accuracy
      alternate model: RFDETRBase(), which matches the DINOv2-compatible configuration
    - resume: add "resume=runs/detect/[output folder]/last.ckpt" to resume training from last checkpoint
    - output_dir: change output directory to "runs/detect/[output folder]" depending on where you want to save the output
    - log_per_class_metrics: if set to True, logs metrics per class, otherwise, logs metrics for all classes combined only
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
        log_per_class_metrics=True,
        early_stopping=True,
        early_stopping_patience=15,
        output_dir="runs/detect/train_rfdetr",
    )

    end = perf_counter()
    training_time = end - start
    Path("runs/detect/train_rfdetr_dinov2/training_time.txt").write_text(
        f"Training time (in seconds): {training_time:.3f} seconds\n"
        f"Training time (in hours): {training_time/3600:.3f} hours\n",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()