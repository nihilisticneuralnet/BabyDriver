# BabyDriver: Real Time Vehicle Detection \& Speed Estimation

A real-time vehicle detection and speed estimation system using **YOLO** model.

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
- **Speed Estimation**: Real-time speed calculation in km/h with perspective correction
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

# Speed calculation parameters
PIXELS_PER_METER = 4       # Calibrate based on your video
SPEED_MULTIPLIER = 6.0     # Fine-tune speed accuracy
FPS = 30                   # Video frame rate

# Input/Output
input_video = "video.mkv"  # Input video path
output_video = "output.mp4" # Output video path
```

### Calibration Guide

1. **PIXELS_PER_METER**: Measure a known distance in your video (e.g., lane width ≈ 3.5m)
   ```python
   PIXELS_PER_METER = measured_pixels / known_meters
   ```

2. **SPEED_MULTIPLIER**: Adjust based on ground truth or expected speeds
   - Start with 1.0 and increase if speeds seem too low
   - Typical range: 3.0-10.0 depending on camera setup


## Future Enhancements

- [ ] **Other Algos**: DeepSORT or ByteTrack integration or Lucas-Kannade, etc for uniform speed estimation


