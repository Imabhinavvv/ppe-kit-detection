from ultralytics import YOLO
import cv2
import cvzone
import math
import os
import sys

def ppe_detection(file_path=None): 
    # Load model with error handling
    model_path = r"D:\IP19\PPE-KIT\ppe.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        sys.exit(1)
    
    model = YOLO(model_path)

    # Video source: Webcam if no file path
    cap = cv2.VideoCapture(0 if file_path is None else file_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {file_path or 'Webcam'}")
        sys.exit(1)

    cap.set(3, 1280)
    cap.set(4, 720)

    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
                  'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

    while True:
        success, img = cap.read()
        if not success:
            print("⚠️ Failed to read frame (end of video or corrupted frame).")
            break

        try:
            results = model(img, stream=True)
        except Exception as e:
            print(f"❌ Model inference error: {e}")
            break

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                conf = float(box.conf[0])
                cls = int(box.cls[0])
                currentClass = classNames[cls] if cls < len(classNames) else "Unknown"

                if conf > 0.5:
                    # Color logic
                    if "NO-" in currentClass:
                        myColor = (0, 0, 255)  # Red for missing equipment
                    elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                        myColor = (0, 255, 0)  # Green for safety equipment
                    else:
                        myColor = (255, 0, 0)  # Blue for other objects

                    # Drawing box and label
                    label = f'{currentClass} {conf:.2f}'
                    cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                       colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        cv2.imshow("PPE Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("✅ Exit requested by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Use video file or set to None to use webcam
    video_file = r"D:/IP19/PPE-KIT/Videos/ppe-1.mp4"
    ppe_detection(video_file)
