# PPE Kit Detection in Automobile Manufacturing

This project focuses on detecting Personal Protective Equipment (PPE) in an automobile manufacturing environment using computer vision techniques.  
By leveraging YOLOv3 and YOLOv8 object detection models, the system identifies the presence or absence of essential safety gear such as helmets, gloves, masks, vests, and shoes in real-time video streams.  
This tool aims to enhance workplace safety by ensuring compliance with PPE protocols.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [License](#license)

## Features

- **Real-Time Detection**: Utilizes YOLOv3 and YOLOv8 models for swift and accurate PPE detection.
- **Multiple Input Sources**: Supports detection from live camera feeds, video files, and static images.
- **Comprehensive PPE Categories**: Detects various PPE items including helmets, gloves, masks, vests, and shoes.
- **Customizable**: Easily adaptable to different environments and additional PPE categories.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Imabhinavvv/ppe-kit-detection.git
   cd ppe-kit-detection
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Detect PPE in a Video File

```bash
python ppe_vid.py --video_path path_to_video.mp4
```

### Detect PPE from a Live Camera Feed

```bash
python ppe_cam.py
```

### Detect PPE in a Static Image

```bash
python detect_gloves.py --image_path path_to_image.jpg
```

### Alternative Detection Script

An alternative script `ppe_alt.py` is available for customized detection scenarios.

## Model Details

- **YOLOv3**:
  - Configuration File: `yolov3.cfg`
  - Weights File: `best.pt`
- **YOLOv8**:
  - Weights File: `yolov8s_custom.pt`
- **Classes File**: `coco.names`

These models have been trained to detect various PPE items relevant to automobile manufacturing settings.

## Project Structure

```
ppe-kit-detection/
├── app.py
├── best.pt
├── checkCam.py
├── coco.names
├── detect_gloves.py
├── main.py
├── output1.mp4
├── ppe.pt
├── ppe_alt.py
├── ppe_cam.py
├── ppe_vid.py
├── requirements.txt
├── yolov3.cfg
├── yolov8s_custom.pt
├── static/
├── templates/
├── videos/
└── README.md
```

- `app.py`: Flask application for web-based interaction.
- `checkCam.py`: Script to verify camera functionality.
- `detect_gloves.py`: Script for detecting gloves in images.
- `main.py`: Main script integrating various functionalities.
- `ppe_cam.py`: Real-time detection from camera feed.
- `ppe_vid.py`: Detection from video files.
- `static/` and `templates/`: Directories for web application assets.
- `videos/`: Directory containing sample videos for testing.

## License

This project is licensed under the [MIT License](LICENSE).

---

For more information and updates, please visit the [GitHub repository](https://github.com/Imabhinavvv/ppe-kit-detection).
