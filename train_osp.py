import ultralytics
import torch
from roboflow import Roboflow

from ultralytics import YOLO

# This checks if ultralytics (and YOLO) is working
ultralytics.checks()

# This checks if CUDA is available
torch.cuda.is_available()

# The Driving Comfort dataset, version 1 of the dataset with no pedestrian label
rf = Roboflow(api_key="IxXkXChvKvzhRnqjmm23")
project = rf.workspace("my-workspace-wvghr").project("-no-peds-driving-comfort-detection")
version = project.version(1)
dataset = version.download("yolov11")

model = YOLO('yolo11n.pt')

'''
model.to('cuda') # Switches YOLO to GPU
'''

train_results = model.train(
    data=f'{dataset.location}/data.yaml',
    epochs=100,
    imgsz=640,
    patience=10
)
