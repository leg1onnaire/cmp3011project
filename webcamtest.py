import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train10/best.pt")  # Update the path if necessary

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam; use another number if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Run inference on the frame
    results = model(frame)

    # results is a list, so we need to access the first element
    result = results[0]

    # Display results on the frame
    frame = result.plot()  # This will overlay the bounding boxes and labels on the frame

    # Show the frame with detections
    cv2.imshow("Webcam Inference", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
