from ultralytics import YOLO
import cv2

# Load the best-performing model obtained during training
model = YOLO("runs/detect/traffic-sign-model/weights/best.pt")

# Load the input image for inference
image_path = "test1.jpg"
image = cv2.imread(image_path)

# Run object detection
result = model(image_path)
print(result)

# Perform inference with a confidence threshold of 0.10
results = model(image_path, conf=0.1)[0]
print(results)

# Iterate through all detected objects
for box in results.boxes:

    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Get the predicted class ID
    cls_id = int(box.cls[0])

    # Get the confidence score
    confidence = float(box.conf[0])

    # Create the label displayed on the image
    label = f"{model.names[cls_id]} | Confidence: {confidence:.2f}"

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the class label and confidence score
    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# Display the prediction result
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the prediction as an output image
cv2.imwrite("prediction_result.jpg", image)
