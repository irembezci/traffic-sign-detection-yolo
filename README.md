# Traffic Sign Detection with YOLOv8

This project demonstrates how to fine-tune a **YOLOv8 Nano** object detection model for traffic sign detection using a custom dataset. Instead of training a model from scratch, a pre-trained YOLOv8 model was adapted to recognize custom traffic sign classes through transfer learning.

The project covers the complete workflow, including dataset preparation, model training, evaluation, inference on unseen images and prediction visualization.

## Features

- Custom traffic sign dataset
- Transfer Learning (Fine-Tuning)
- YOLOv8 Nano
- Model training
- Model evaluation
- Inference on unseen images
- Bounding box visualization
- Confidence score visualization
- False positive analysis

# Project Structure

```text
traffic-sign-detection-yolo/
│
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── runs/
│   └── detect/
│
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

# Dataset Preparation

The dataset follows the standard YOLO directory structure.

```text
dataset/
    train/
        images/
        labels/

    valid/
        images/
        labels/

    test/
        images/
        labels/
```

Each image has a corresponding annotation file (`.txt`) containing the object class and normalized bounding box coordinates.

Example annotation:

```text
0 0.53 0.42 0.18 0.22
```

Where:

- Class ID
- X center
- Y center
- Width
- Height

All coordinates are normalized.

# Configure `data.yaml`

```yaml
train: dataset/train/images
val: dataset/valid/images
test: dataset/test/images

nc: 15

names:
  0: bend right
  1: hump
  ...
```

| Parameter | Description |
|-----------|-------------|
| train | Training image directory |
| val | Validation image directory |
| test | Test image directory |
| nc | Number of classes |
| names | Class labels |

# Training Workflow

## Step 1 — Load the Pre-trained Model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

Instead of training a model from scratch, YOLOv8 Nano starts from weights already trained on the COCO dataset. The model is then fine-tuned on the custom traffic sign dataset.

## Step 2 — Train the Model

```python
model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    name="traffic-sign-model"
)
```

During each epoch the model performs the following operations:

```text
Training Images
      │
      ▼
Forward Pass
      │
      ▼
Prediction
      │
      ▼
Loss Calculation
      │
      ▼
Backpropagation
      │
      ▼
Weight Update
```

After every epoch the model is evaluated using the validation dataset.

# Understanding Training Parameters

| Parameter | Description |
|-----------|-------------|
| epochs | Number of complete passes through the training dataset. |
| imgsz | Input image size. |
| batch | Number of images processed before updating the weights. |
| data | Dataset configuration file. |
| name | Folder where the training results are saved. |

During this project the model was trained using different epoch values (5, 10 and 20) to compare the results.

# Monitoring Training

YOLO reports three training losses.

## Box Loss

Measures how accurately the predicted bounding boxes match the ground truth.

Lower values indicate better localization.

## Classification Loss

Measures how accurately the detected object is classified.

Lower values indicate better classification.

## DFL Loss

Distribution Focal Loss improves bounding box localization accuracy.

# Validation Metrics

The model is automatically evaluated after each epoch.

| Metric | Description |
|---------|-------------|
| Precision | Percentage of correct detections |
| Recall | Percentage of detected ground-truth objects |
| mAP50 | Average Precision at IoU = 0.50 |
| mAP50-95 | Average Precision across multiple IoU thresholds |

# Loading the Best Model

YOLO automatically saves two checkpoints.

- **best.pt** → Best validation performance
- **last.pt** → Last completed epoch

```python
model = YOLO("runs/detect/traffic-sign-model/weights/best.pt")
```

# Testing the Model

The trained model was evaluated on three unseen traffic sign images.

```python
results = model(image_path, conf=0.25)[0]
```

To analyze low-confidence detections, inference was also performed using:

```python
results = model(image_path, conf=0.10)[0]
```

# Visualizing Predictions

Each prediction contains:

- Bounding box
- Class label
- Confidence score

The predictions were drawn on the original image using OpenCV.

# Sample Predictions

## Test Image 1

| Original | Prediction |
|----------|------------|
| ![](images/test1.jpg) | ![](images/prediction1.jpg) |

Correctly detected the traffic sign with a high confidence score.

## Test Image 2

| Original | Prediction |
|----------|------------|
| ![](images/test2.jpg) | ![](images/prediction2.jpg) |

Correctly detected one traffic sign but also produced a false positive by classifying a truck windshield as a parking sign.

## Test Image 3

| Original | Prediction |
|----------|------------|
| ![](images/test3.jpg) | ![](images/prediction3.jpg) |

Successfully detected the traffic sign under different viewing conditions.

# Experimental Observations

During this project several experiments were performed.

- Compared training using 5, 10 and 20 epochs.
- Tested different confidence thresholds.
- Evaluated the model on unseen images.
- Observed false positives and small-object detection challenges.

# Challenges Encountered

The following issues were encountered and resolved during development:

- Virtual environment configuration
- Missing Python packages
- Incorrect image path definitions
- Tensor indexing (`KeyError`)
- Confidence threshold tuning
- Small object detection
- False positive predictions

# Conclusion

This project demonstrates an end-to-end object detection workflow using YOLOv8. A pre-trained YOLOv8 Nano model was successfully fine-tuned on a custom traffic sign dataset, evaluated on unseen images and analyzed using different confidence thresholds.

The project also highlights practical challenges such as false positives, confidence tuning and small object detection, providing a realistic understanding of deploying custom object detection models.
