"""
Traffic Sign Detection using YOLOv8

Project Workflow

1. Collect a traffic sign dataset
2. Configure the dataset for YOLO
3. Train a custom YOLOv8 model
4. Evaluate the trained model
5. Perform inference on unseen images

Why YOLO?

YOLO (You Only Look Once) is a real-time object detection algorithm that
predicts object classes and bounding boxes in a single forward pass,
making it suitable for applications such as autonomous driving,
traffic sign recognition and intelligent transportation systems.

Dataset

Traffic Sign Detection YOLOv8 Dataset
Source: Roboflow Universe
"""

from ultralytics import YOLO

# Load the pre-trained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Fine-tune the model using the custom traffic sign dataset
model.train(
    data="traffic-sign-detection-dataset/data.yaml",   # Dataset configuration file
    epochs=2,                                          # Number of training epochs
    imgsz=640,                                         # Input image size
    batch=8,                                           # Number of images processed per batch
    name="traffic-sign-model",                         # Output folder name
    lr0=0.01,                                          # Initial learning rate
    optimizer="SGD",                                   # Optimization algorithm
    weight_decay=0.0005,                               # Regularization to reduce overfitting
    momentum=0.935,                                    # Momentum used by the SGD optimizer
    patience=50,                                       # Early stopping patience
    workers=2,                                         # Number of data loading workers
    device="cpu",                                      # Training device (CPU or CUDA)
    save=True,                                         # Save model checkpoints
    save_period=1,                                     # Save a checkpoint after every epoch
    val=True,                                          # Run validation after each epoch
    verbose=True                                       # Display detailed training logs
)
