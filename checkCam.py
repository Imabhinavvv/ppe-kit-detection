import cv2

cap = cv2.VideoCapture(0)  # 0 is usually the default camera
if not cap.isOpened():
    print("Error: Camera not found or could not be opened.")
else:
    print("Camera is working!")
    cap.release()
