# BabyDriver: Real Time Vehicle Detection & Speed Estimation
A real-time vehicle detection and speed estimation system using **YOLO** model with **optical flow** tracking and **homography-based** speed calculation.

## Example Demo
<p align="center">
  <img src="/demo/output1.gif" width="250"/>
  <img src="/demo/output2.gif" width="250"/>
  <img src="/demo/output3.gif" width="250"/>
</p>
<!--https://github.com/user-attachments/assets/67ee68e6-9de8-4d2d-b4d2-adc35fd6ef0a-->
<!--https://github.com/user-attachments/assets/3c293acc-7d77-46d1-8d23-f27892331525-->

## Features
- **Multi-Vehicle Detection**: Detects cars, motorcycles, buses, and trucks using YOLOv3
- **Object Tracking**: Persistent tracking across frames with unique IDs
- **Optical Flow Speed Estimation**: Lucas-Kanade optical flow within bounding boxes for accurate motion tracking
- **Homography-Based Calculation**: Geometrically correct speed estimation using ground-plane projection
- **Speed Violation Detection**: Visual alerts for vehicles exceeding 80 km/h

## Installation

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nihilisticneuralnet/BabyDriver.git
   cd BabyDriver
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv3 model files**
   ```bash
   # Download YOLOv3 weights (248 MB)
   wget https://pjreddie.com/media/files/yolov3.weights -O yolov3-320.weights
   
   # Download YOLOv3 config
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolov3-320.cfg
   
   # Download COCO class names
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O coco.names
   ```

4. **Prepare your video**
   ```bash
   # Place your input video as 'video.mkv' in the project directory
   # Or modify the video path in main.py
   ```

## Usage

### Basic Usage
```bash
python main.py
```

### Custom Configuration
Modify these parameters in `main.py` for your specific use case:

```python
# Detection parameters
confThreshold = 0.2        # Confidence threshold (0.1-0.9)
nmsThreshold = 0.2         # Non-max suppression threshold

# Optical flow parameters
lk_params = dict(winSize=(21, 21), maxLevel=3, ...)
feature_params = dict(maxCorners=50, qualityLevel=0.01, ...)

# Input/Output
input_video = "video.mkv"      # Input video path
output_video = "output.mp4"    # Output video path
auto_calibrate = True          # Automatic vs manual homography
```

### Calibration Guide

1. **Automatic Calibration** (Default):
   ```python
   detector.processVideo("output.mp4", auto_calibrate=True)
   ```
   Uses default road assumptions (10m × 15m rectangular area)

2. **Manual Calibration** (Recommended for accuracy):
   ```python
   detector.processVideo("output.mp4", auto_calibrate=False)
   ```
   - Click 4 points on the road plane (e.g., corners of parking spot)
   - Enter real-world coordinates in meters for each point
   - System computes homography matrix automatically



## What not worked

- Farneback for object tracking was bad


