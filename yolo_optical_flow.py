import cv2
import numpy as np
import math
import time
import matplotlib
matplotlib.use('Agg')
cv2.setNumThreads(4)

cap = cv2.VideoCapture("video.mkv")
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
# IMPORTANT: You must calibrate this for your video!
# Select 4 points on the road that form a known rectangle in real world
# Example: corners of a parking spot or road marking of known dimensions
# 
# To calibrate:
# 1. Pause video and identify 4 coplanar points on the road
# 2. Measure their real-world coordinates (in meters, origin at one corner)
# 3. Get their pixel coordinates in the image
# 4. Compute H = cv2.getPerspectiveTransform(src_points, dst_points)
#
# Example calibration (REPLACE WITH YOUR VALUES):
# src_points: pixel coordinates of 4 road points in image
# dst_points: real-world coordinates of same 4 points in meters
#
# For demonstration, using identity (you MUST replace this):
H_MATRIX = None  # Will be set during video processing with user-defined points

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# goodFeaturesToTrack parameters
feature_params = dict(maxCorners=30,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

class tracker:
    def __init__(self):
        self.id_count = 0
        self.center_points = {}
        self.disappeared = {}
        self.max_disappeared = 10
    
    def update(self, objects_rect):
        if len(objects_rect) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []
        
        objects_bbs_ids = []
        
        if len(self.center_points) == 0:
            for rect in objects_rect:
                x, y, w, h, index = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1
        else:
            input_centroids = []
            for rect in objects_rect:
                x, y, w, h, index = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                input_centroids.append((cx, cy, x, y, w, h, index))
            
            object_ids = list(self.center_points.keys())
            object_centroids = list(self.center_points.values())
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array([(c[0], c[1]) for c in input_centroids]), axis=2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] <= 50:
                    object_id = object_ids[row]
                    self.center_points[object_id] = (input_centroids[col][0], input_centroids[col][1])
                    objects_bbs_ids.append([input_centroids[col][2], input_centroids[col][3], 
                                          input_centroids[col][4], input_centroids[col][5], 
                                          object_id, input_centroids[col][6]])
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
                    
                    if object_id in self.disappeared:
                        del self.disappeared[object_id]
            
            unused_row_indices = set(range(0, D.shape[0])) - used_row_indices
            unused_col_indices = set(range(0, D.shape[1])) - used_col_indices
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] = self.disappeared.get(object_id, 0) + 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    cx, cy, x, y, w, h, index = input_centroids[col]
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                    self.id_count += 1
        
        return objects_bbs_ids
    
    def deregister(self, object_id):
        del self.center_points[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

class detect:
    def __init__(self):
        self.boxes = []
        self.classIds = []
        self.confidence_scores = []
        self.detection = []
        self.track = tracker()
        
        # Optical flow state
        self.prev_gray = None
        self.object_features = {}  # object_id -> feature points (Nx1x2)
        self.prev_time = {}
        self.speed_history = {}  # For temporal smoothing (EMA)
        self.min_features_threshold = 5  # Re-initialize if below this
        self.frame_count = 0
        
    def initialize_features(self, gray, x, y, w, h):
        """Initialize feature points within bounding box using goodFeaturesToTrack"""
        # Create mask for the bounding box region
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        
        # Detect features
        features = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
        
        return features
    
    def apply_homography_to_points(self, points, H):
        """Project points from image space to ground plane using homography"""
        if points is None or len(points) == 0 or H is None:
            return None
        
        # Reshape points for cv2.perspectiveTransform
        # Input: Nx1x2, Output: Nx1x2 in world coordinates
        world_points = cv2.perspectiveTransform(points.astype(np.float32), H)
        
        return world_points
    
    def calculate_speed_optical_flow(self, object_id, gray, x, y, w, h, current_time):
        """
        Calculate speed using optical flow within bounding box and homography
        Returns speed in km/h
        """
        global H_MATRIX
        
        # If no homography matrix, cannot compute speed
        if H_MATRIX is None:
            return 0.0
        
        # Initialize features for new objects or when count is too low
        if (object_id not in self.object_features or 
            self.object_features[object_id] is None or
            len(self.object_features[object_id]) < self.min_features_threshold):
            
            self.object_features[object_id] = self.initialize_features(gray, x, y, w, h)
            self.prev_time[object_id] = current_time
            return 0.0
        
        # Need previous frame to compute optical flow
        if self.prev_gray is None:
            self.prev_time[object_id] = current_time
            return 0.0
        
        # Track features using Lucas-Kanade
        prev_features = self.object_features[object_id]
        
        next_features, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_features, None, **lk_params)
        
        # Filter good points: status=1, low error, inside bounding box
        good_prev = []
        good_next = []
        
        if next_features is not None:
            for i, (st, err) in enumerate(zip(status, error)):
                if st == 1 and err < 50:  # Error threshold
                    px, py = next_features[i].ravel()
                    # Check if point is still within bounding box (with margin)
                    margin = 5
                    if (x - margin <= px <= x + w + margin and 
                        y - margin <= py <= y + h + margin):
                        good_prev.append(prev_features[i])
                        good_next.append(next_features[i])
        
        # If too few good points, re-initialize
        if len(good_next) < self.min_features_threshold:
            self.object_features[object_id] = self.initialize_features(gray, x, y, w, h)
            self.prev_time[object_id] = current_time
            return self.get_smoothed_speed(object_id, 0.0)
        
        # Convert to numpy arrays
        good_prev = np.array(good_prev).reshape(-1, 1, 2)
        good_next = np.array(good_next).reshape(-1, 1, 2)
        
        # Update features for next iteration
        self.object_features[object_id] = good_next
        
        # Project to world coordinates
        world_prev = self.apply_homography_to_points(good_prev, H_MATRIX)
        world_next = self.apply_homography_to_points(good_next, H_MATRIX)
        
        if world_prev is None or world_next is None:
            return self.get_smoothed_speed(object_id, 0.0)
        
        # Calculate 2D displacements in world coordinates (meters)
        displacements = world_next - world_prev
        displacement_magnitudes = np.sqrt(displacements[:, 0, 0]**2 + displacements[:, 0, 1]**2)
        
        # Use median displacement to suppress outliers
        median_displacement = np.median(displacement_magnitudes)
        
        # Calculate time difference
        time_diff = current_time - self.prev_time.get(object_id, current_time)
        self.prev_time[object_id] = current_time
        
        if time_diff <= 0 or time_diff > 1.0:  # Sanity check
            return self.get_smoothed_speed(object_id, 0.0)
        
        # Speed in m/s
        speed_ms = median_displacement / time_diff
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        # Sanity bounds
        speed_kmh = max(0, min(speed_kmh, 200))
        
        # Apply temporal smoothing
        return self.get_smoothed_speed(object_id, speed_kmh)
    
    def get_smoothed_speed(self, object_id, new_speed):
        """Apply exponential moving average (EMA) for temporal smoothing"""
        alpha = 0.3  # Smoothing factor (0 = no update, 1 = no smoothing)
        
        if object_id not in self.speed_history:
            self.speed_history[object_id] = new_speed
            return new_speed
        
        # EMA formula: S_t = alpha * x_t + (1 - alpha) * S_{t-1}
        smoothed = alpha * new_speed + (1 - alpha) * self.speed_history[object_id]
        self.speed_history[object_id] = smoothed
        
        return smoothed
    
    def cleanup_object(self, object_id):
        """Remove optical flow state when object is deregistered"""
        if object_id in self.object_features:
            del self.object_features[object_id]
        if object_id in self.prev_time:
            del self.prev_time[object_id]
        if object_id in self.speed_history:
            del self.speed_history[object_id]
        
    def postProcess(self, outputs, img, gray):
        global detected_classNames
        
        self.boxes = []
        self.classIds = []
        self.confidence_scores = []
        self.detection = []
        
        height, width = img.shape[:2]
        current_time = time.time()
        self.frame_count += 1

        for output in outputs:
            for det in output:
                scores = det[5:]
                self.classId = np.argmax(scores)
                confidence = scores[self.classId]
                if self.classId in required_class_index:
                    if confidence > confThreshold:
                        w, h = int(det[2] * width), int(det[3] * height)
                        x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                        self.boxes.append([x, y, w, h])
                        self.classIds.append(self.classId)
                        self.confidence_scores.append(float(confidence))

        # Non-max suppression
        if len(self.boxes) > 0:
            indices = cv2.dnn.NMSBoxes(self.boxes, self.confidence_scores, confThreshold, nmsThreshold)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = self.boxes[i][0], self.boxes[i][1], self.boxes[i][2], self.boxes[i][3]
                    self.detection.append([x, y, w, h, required_class_index.index(self.classIds[i])])

        # Track existing objects and clean up disappeared ones
        current_ids = set()
        
        if len(self.detection) > 0:
            boxes_ids = self.track.update(self.detection)
            
            for box_id in boxes_ids:
                x, y, w, h, object_id, index = box_id
                current_ids.add(object_id)
                
                # Calculate speed using optical flow and homography
                speed_kmh = self.calculate_speed_optical_flow(object_id, gray, x, y, w, h, current_time)
                
                color = [int(c) for c in colors[self.classIds[index] if index < len(self.classIds) else 0]]
                name = classNames[self.classIds[index] if index < len(self.classIds) else 0]
                
                # Red if > 80 km/h, green otherwise
                speed_color = (0, 0, 255) if speed_kmh > 80 else (0, 255, 0)
                
                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Draw class label and confidence
                cv2.putText(img, f'{name.upper()} {int(self.confidence_scores[index] * 100)}%' if index < len(self.confidence_scores) else f'{name.upper()}',
                           (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw speed (only if meaningful)
                if speed_kmh > 5:  # Threshold to avoid displaying noise
                    cv2.putText(img, f"Speed: {speed_kmh:.1f} km/h", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 1)
                
                # Draw object ID
                cv2.putText(img, f"ID: {object_id}", (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Visualize tracked features (optional, for debugging)
                if object_id in self.object_features and self.object_features[object_id] is not None:
                    for pt in self.object_features[object_id]:
                        px, py = pt.ravel()
                        cv2.circle(img, (int(px), int(py)), 2, (0, 255, 255), -1)
        
        # Clean up features for disappeared objects
        all_ids = set(self.object_features.keys())
        disappeared_ids = all_ids - current_ids
        for obj_id in disappeared_ids:
            self.cleanup_object(obj_id)
        
        # Update previous gray frame
        self.prev_gray = gray.copy()

    def setup_homography(self, frame):
        """
        Interactive setup of homography matrix
        User clicks 4 points on the road and provides real-world coordinates
        """
        global H_MATRIX
        
        print("\n" + "="*60)
        print("HOMOGRAPHY CALIBRATION")
        print("="*60)
        print("You need to select 4 points on the road plane that form a rectangle")
        print("in the real world (e.g., corners of a parking spot or road marking)")
        print("\nInstructions:")
        print("1. Click 4 points in order (clockwise from top-left recommended)")
        print("2. After 4 clicks, you'll be asked for real-world coordinates")
        print("3. Provide coordinates in meters (e.g., 0,0 for origin, 3,0 for 3m right)")
        print("="*60 + "\n")
        
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame_copy, f"P{len(points)}", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Calibration", frame_copy)
        
        frame_copy = frame.copy()
        cv2.imshow("Calibration", frame_copy)
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        print("Click 4 points on the road... (Press any key after 4 points)")
        cv2.waitKey(0)
        cv2.destroyWindow("Calibration")
        
        if len(points) != 4:
            print(f"Error: Need exactly 4 points, got {len(points)}")
            print("Using default homography (identity - speeds will be incorrect!)")
            H_MATRIX = np.eye(3, dtype=np.float32)
            return
        
        print(f"\nImage points selected: {points}")
        print("\nNow enter the real-world coordinates (in meters) for each point:")
        
        world_points = []
        for i in range(4):
            while True:
                try:
                    coords = input(f"Point {i+1} real-world coords (x,y in meters, e.g., '0,0'): ")
                    x, y = map(float, coords.split(','))
                    world_points.append([x, y])
                    break
                except:
                    print("Invalid format. Use: x,y (e.g., '3.5,2.0')")
        
        src_pts = np.array(points, dtype=np.float32)
        dst_pts = np.array(world_points, dtype=np.float32)
        
        H_MATRIX = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        print("\n" + "="*60)
        print("Homography matrix computed successfully!")
        print("="*60)
        print(H_MATRIX)
        print("="*60 + "\n")

    def processVideo(self, output_path="output.mp4"):
        """Process video and save output"""
        global FPS, H_MATRIX
        
        FPS = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        new_width = int(width * 0.5)
        new_height = int(height * 0.5)
        
        print(f"Processing video: {width}x{height} -> {new_width}x{new_height}")
        print(f"FPS: {FPS}, Total frames: {total_frames}")
        
        # Capture first frame for homography calibration
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Cannot read video")
            return
        
        first_frame_resized = cv2.resize(first_frame, (new_width, new_height))
        
        # Setup homography
        self.setup_homography(first_frame_resized)
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, FPS, (new_width, new_height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (new_width, new_height))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
                
                net.setInput(blob)
                layersNames = net.getLayerNames()
                outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)
                
                self.postProcess(outputs, frame, gray)
                
                out.write(frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
                
        except Exception as e:
            print(f"Error during video processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            cap.release()
            out.release()
            print(f"Video processing complete! Output saved to: {output_path}")
            print(f"Total frames processed: {frame_count}")

detector = detect()
detector.processVideo("output.mp4")