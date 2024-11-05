from ultralytics import YOLO
import cv2

# Load a COCO-pretrained YOLOv11 model
model = YOLO("yolo11n.pt")  # Make sure "yolo11n.pt" is downloaded and in the same directory or specify the full path

# Initialize the two cameras
cap1 = cv2.VideoCapture(0)  # First camera
cap2 = cv2.VideoCapture(1)  # Second camera

# Check if both cameras opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

def detect_objects(frame):
    """Runs YOLOv11 detection on the given frame and returns the results."""
    results = model(frame)
    return results

while True:
    # Capture frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Could not read frame from one or both cameras.")
        break

    # Run YOLOv11 model on both frames
    results1 = detect_objects(frame1)
    results2 = detect_objects(frame2)

    # Flag to check if any object is detected in either camera
    found_object = False

    # Process detections for Camera 1
    for result in results1[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        label = model.names[int(result.cls)]
        confidence = result.conf[0]
        cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame1, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        found_object = True

    # Process detections for Camera 2
    for result in results2[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        label = model.names[int(result.cls)]
        confidence = result.conf[0]
        cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame2, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        found_object = True

    # Display frames with detections
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    # Print detection status
    if found_object:
        print("Object detected in one or both cameras.")
    else:
        print("No object detected. Waiting...")

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close all OpenCV windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
