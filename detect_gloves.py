import cv2
from ultralytics import YOLO
import random
from tensorflow.keras.layers import Conv2D

# Load video file
cap = cv2.VideoCapture('D:/InfosysSpringboard/ppeKitDetection/Videos/ppe-2.mp4')
model = YOLO('D:/InfosysSpringboard/ppeKitDetection/best.pt')

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
size = (frame_width, frame_height)

# Initialize video writer
video = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

# Function to plot bounding boxes
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.7)
    
    # Static zones for visualization
    plot_one_box((0, 860, 1200, 250), frame, color=(128, 0, 128), label='Height operation', line_thickness=3)
    plot_one_box((1700, 670, 2100, 1900), frame, color=(128, 128, 0), label='Deflashing station', line_thickness=3)
    
    for f in results:
        boxes = f.boxes.xyxy
        category = f.boxes.cls
        confidence = f.boxes.conf
        
        for i, j, k in zip(boxes, category, confidence):
            xmin, ymin, xmax, ymax = int(i[0]), int(i[1]), int(i[2]), int(i[3])
            
            labels = ['withHelmet', 'withoutHelmet', 'withGoggle', 'withoutGoggle', 'withGlove', 
                      'withoutGlove', 'withoutShoe', 'withShoe', 'Mobile']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                      (255, 0, 255), (0, 255, 255), (255, 128, 0), (255, 0, 128), (128, 0, 255)]
            
            if 0 <= j < len(labels):
                plot_one_box((xmin, ymin, xmax, ymax), frame, color=colors[int(j)], label=labels[int(j)], line_thickness=3)

    # Write the processed frame to the output video
    video.write(frame)

    # Display the frame in a pop-up window
    cv2.imshow('PPE Detection Output', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
video.release()
cv2.destroyAllWindows()
