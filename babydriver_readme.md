# BabyDriver: Real Time Vehicle Detection & Speed Estimation
A real-time vehicle detection and speed estimation system using **YOLO** model with **Lucas-Kanade optical flow** and **homography-based speed calculation**.

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
- **Optical Flow Tracking**: Lucas-Kanade optical flow for precise motion estimation within bounding boxes
- **Homography-Based Speed Estimation**: Geometrically correct speed calculation using ground plane projection
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
lk_params = dict(winSize=(15, 15), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=30, qualityLevel=0.3, 
                      minDistance=7, blockSize=7)

# Input/Output
input_video = "video.mkv"  # Input video path
output_video = "output.mp4" # Output video path
```

### Calibration Guide
On first run, the system will pause and display the first frame for homography calibration:

1. **Click 4 points on the road** that form a rectangle in the real world (e.g., corners of a parking spot or road marking)
   - Click points in order (clockwise from top-left recommended)
   - Points must be coplanar on the road surface

2. **Enter real-world coordinates** for each point when prompted
   ```
   Point 1 real-world coords (x,y in meters, e.g., '0,0'): 0,0
   Point 2 real-world coords (x,y in meters, e.g., '0,0'): 3,0
   Point 3 real-world coords (x,y in meters, e.g., '0,0'): 3,4
   Point 4 real-world coords (x,y in meters, e.g., '0,0'): 0,4
   ```

3. **Processing begins automatically** with geometrically correct speed estimation

The homography matrix maps image pixels to real-world ground plane coordinates, eliminating the need for manual calibration of pixels-per-meter or speed multipliers.