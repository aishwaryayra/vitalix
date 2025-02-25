from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize video capture
cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # Use video file

# Load YOLO model
model = YOLO("../models/yolov8n.pt")

# Class names
classNames = ["fake", "real"]

# Initialize FPS calculation
prev_frame_time = 0

while True:
    # Read a frame from the video
    success, img = cap.read()
    if not success:
        break  # Exit loop if the video ends

    # Get YOLO model results
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Draw bounding box
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])

            # Draw text on image
            cvzone.putTextRect(img, f'{classNames[cls]}: ({max(0, x1)}, {max(35, y1)})', scale=1, thickness=2)

    # Calculate FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0
    prev_frame_time = new_frame_time
    print(f'FPS: {fps:.2f}')

    # Show the image
    cv2.imshow("Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
