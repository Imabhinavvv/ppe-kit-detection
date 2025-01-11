## Video without alert sound

from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize video capture (Camera or Video)
cap = cv2.VideoCapture(0)  # Default to camera
# Comment the above line and uncomment the below one for video file
# cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video

# Set frame width and height (only for camera)
if cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 0:
    cap.set(3, 1280)  # Frame width
    cap.set(4, 720)   # Frame height

# Check if camera/video opened successfully
if not cap.isOpened():
    print("Error: Camera or video file could not be opened.")
    exit()

# Load YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")
# model = YOLO("../Yolo-Weights/yolov8n.pt")
model = YOLO("best.pt")  # Your custom-trained model

# Define class names
classNames = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle', 'Gloves'
]

# Initialize FPS variables
prev_frame_time = 0
new_frame_time = 0
fps = 0  # Initial FPS

while True:
    success, img = cap.read()  # Capture frame-by-frame

    if not success:  # Handle frame capture failure
        print("Error: Failed to capture frame. Ending program.")
        break

    # YOLO detection
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cls = int(box.cls[0])
            if classNames[cls] in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
               color = (0, 0, 255)  # Red
            else:
               color = (0, 255, 0)  # Green


            # Draw rectangle
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Extract confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Display class name and confidence
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate and display FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps = int(fps)  # Convert to integer
    prev_frame_time = new_frame_time

    # Display FPS
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)
    print(f'FPS: {fps}')

    # Show image
    cv2.imshow("Image", img)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera/video and close windows
cap.release()
cv2.destroyAllWindows()
