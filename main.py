import cv2
import numpy as np
import math
import time
import matplotlib
matplotlib.use('Agg')
from detect import detect

cv2.setNumThreads(4)

cap = cv2.VideoCapture("*.mkv")
input_size = 320

confThreshold = 0.2
nmsThreshold = 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

FPS = 30 

classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
required_class_index = [2, 3, 5, 7]

detected_classNames = []

modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

# Homography matrix H: image -> ground plane (meters)
H_MATRIX = None

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# goodFeaturesToTrack parameters
feature_params = dict(maxCorners=50,
                      qualityLevel=0.01,
                      minDistance=5,
                      blockSize=7)


def processVideo(output_path="output.mp4"):
    global FPS, PIXELS_PER_METER
    
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    PIXELS_PER_METER = max(2, width // 400)
    
    new_width = int(width * 0.5)
    new_height = int(height * 0.5)
    
    print(f"Processing video: {width}x{height} -> {new_width}x{new_height}")
    print(f"FPS: {FPS}, Total frames: {total_frames}")
    print(f"Estimated pixels per meter: {PIXELS_PER_METER}")
    print(f"Speed multiplier: {SPEED_MULTIPLIER}")
    print("Note: Adjust PIXELS_PER_METER and SPEED_MULTIPLIER if speeds seem unrealistic")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (new_width, new_height))
    
    detector = detect(net, classNames, colors, required_class_index, confThreshold, nmsThreshold, PIXELS_PER_METER, SPEED_MULTIPLIER)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (new_width, new_height))
            
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
            
            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(outputNames)
            
            detector.postProcess(outputs, frame)
            
            # cv2.imshow('Video', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
            
    except Exception as e:
        print(f"Error during video processing: {e}")
    
    finally:
        cap.release()
        out.release()
        # cv2.destroyAllWindows()
        print(f"Video processing complete! Output saved to: {output_path}")
        print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    detector = detect()
    # Set auto_calibrate=False for manual calibration
    detector.processVideo("output.mp4", auto_calibrate=True)

