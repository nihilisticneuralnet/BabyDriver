import cv2
import numpy as np
import math
import time
from track import tracker

def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

class detect:
    def __init__(self):
        self.boxes = []
        self.classIds = []
        self.confidence_scores = []
        self.detection = []
        self.track = tracker()
        
        # Optical flow state
        self.prev_gray = None
        self.object_features = {}
        self.prev_time = {}
        self.speed_history = {}
        self.min_features_threshold = 3
        self.frame_count = 0
        self.frame_time = None
        
    def initialize_features(self, gray, x, y, w, h):
        """Initialize feature points within bounding box"""
        # Add padding to bounding box but keep within image bounds
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)
        
        # Create mask
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        
        # Detect features
        features = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
        
        return features
    
    def apply_homography_to_points(self, points, H):
        """Project points from image to world coordinates"""
        if points is None or len(points) == 0 or H is None:
            return None
        
        world_points = cv2.perspectiveTransform(points.astype(np.float32), H)
        return world_points
    
    def calculate_speed_optical_flow(self, object_id, gray, x, y, w, h):
        """Calculate speed using optical flow and homography"""
        global H_MATRIX
        
        # Check if we have homography
        if H_MATRIX is None:
            return -1.0, "NO_HOMOGRAPHY"
        
        # Initialize features for new objects
        if (object_id not in self.object_features or 
            self.object_features[object_id] is None or
            len(self.object_features[object_id]) < self.min_features_threshold):
            
            features = self.initialize_features(gray, x, y, w, h)
            self.object_features[object_id] = features
            self.prev_time[object_id] = self.frame_time
            
            if features is None or len(features) < self.min_features_threshold:
                return -1.0, "NO_FEATURES_INIT"
            return 0.0, "INIT"
        
        # Need previous frame
        if self.prev_gray is None:
            self.prev_time[object_id] = self.frame_time
            return 0.0, "NO_PREV_FRAME"
        
        # Track features
        prev_features = self.object_features[object_id]
        
        if prev_features is None or len(prev_features) == 0:
            return -1.0, "NO_PREV_FEATURES"
        
        next_features, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_features, None, **lk_params)
        
        # Filter good points
        good_prev = []
        good_next = []
        
        if next_features is not None and status is not None:
            for i, (st, err) in enumerate(zip(status, error)):
                if st == 1 and err < 50:
                    px, py = next_features[i].ravel()
                    # Check if still in bounding box (with margin)
                    margin = 20
                    if (x - margin <= px <= x + w + margin and 
                        y - margin <= py <= y + h + margin):
                        good_prev.append(prev_features[i])
                        good_next.append(next_features[i])
        
        # Re-initialize if too few points
        if len(good_next) < self.min_features_threshold:
            features = self.initialize_features(gray, x, y, w, h)
            self.object_features[object_id] = features
            self.prev_time[object_id] = self.frame_time
            return 0.0, f"REINIT({len(good_next)}pts)"
        
        # Convert to arrays
        good_prev = np.array(good_prev).reshape(-1, 1, 2)
        good_next = np.array(good_next).reshape(-1, 1, 2)
        
        # Update features
        self.object_features[object_id] = good_next
        
        # Project to world coordinates
        world_prev = self.apply_homography_to_points(good_prev, H_MATRIX)
        world_next = self.apply_homography_to_points(good_next, H_MATRIX)
        
        if world_prev is None or world_next is None:
            return -1.0, "HOMOGRAPHY_FAILED"
        
        # Calculate displacements in world space (meters)
        displacements = world_next - world_prev
        displacement_magnitudes = np.sqrt(displacements[:, 0, 0]**2 + displacements[:, 0, 1]**2)
        
        # Use median to suppress outliers
        median_displacement = np.median(displacement_magnitudes)
        
        # Time difference (use frame rate for consistent timing)
        prev_time = self.prev_time.get(object_id, self.frame_time - 1.0/FPS)
        time_diff = self.frame_time - prev_time
        self.prev_time[object_id] = self.frame_time
        
        if time_diff <= 0 or time_diff > 1.0:
            return 0.0, "BAD_TIME"
        
        # Speed calculation
        speed_ms = median_displacement / time_diff
        speed_kmh = speed_ms * 3.6
        
        # Bounds
        speed_kmh = max(0, min(speed_kmh, 200))
        
        # Apply smoothing
        smoothed_speed = self.get_smoothed_speed(object_id, speed_kmh)
        
        return smoothed_speed, f"OK({len(good_next)}pts,{median_displacement:.3f}m)"
    
    def get_smoothed_speed(self, object_id, new_speed):
        """Exponential moving average"""
        alpha = 0.4
        
        if object_id not in self.speed_history:
            self.speed_history[object_id] = new_speed
            return new_speed
        
        smoothed = alpha * new_speed + (1 - alpha) * self.speed_history[object_id]
        self.speed_history[object_id] = smoothed
        
        return smoothed
    
    def cleanup_object(self, object_id):
        """Remove state for deregistered objects"""
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

        # NMS
        if len(self.boxes) > 0:
            indices = cv2.dnn.NMSBoxes(self.boxes, self.confidence_scores, confThreshold, nmsThreshold)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = self.boxes[i][0], self.boxes[i][1], self.boxes[i][2], self.boxes[i][3]
                    self.detection.append([x, y, w, h, required_class_index.index(self.classIds[i])])

        current_ids = set()
        
        if len(self.detection) > 0:
            boxes_ids = self.track.update(self.detection)
            
            for box_id in boxes_ids:
                x, y, w, h, object_id, index = box_id
                current_ids.add(object_id)
                
                # Calculate speed
                speed_kmh, status = self.calculate_speed_optical_flow(object_id, gray, x, y, w, h)
                
                color = [int(c) for c in colors[self.classIds[index] if index < len(self.classIds) else 0]]
                name = classNames[self.classIds[index] if index < len(self.classIds) else 0]
                
                # Color based on speed
                if speed_kmh >= 0:
                    speed_color = (0, 0, 255) if speed_kmh > 80 else (0, 255, 0)
                else:
                    speed_color = (128, 128, 128)
                
                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                cv2.putText(img, f'{name.upper()} {int(self.confidence_scores[index] * 100)}%' if index < len(self.confidence_scores) else f'{name.upper()}',
                           (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw speed - ALWAYS show it with status
                if speed_kmh >= 0:
                    speed_text = f"{speed_kmh:.1f} km/h"
                else:
                    speed_text = status
                    
                cv2.putText(img, speed_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 1)
                
                # Draw ID
                cv2.putText(img, f"ID: {object_id}", (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw tracked features
                if object_id in self.object_features and self.object_features[object_id] is not None:
                    for pt in self.object_features[object_id]:
                        px, py = pt.ravel()
                        cv2.circle(img, (int(px), int(py)), 2, (0, 255, 255), -1)
        
        # Cleanup disappeared objects
        all_ids = set(self.object_features.keys())
        disappeared_ids = all_ids - current_ids
        for obj_id in disappeared_ids:
            self.cleanup_object(obj_id)
        
        # Update previous frame
        self.prev_gray = gray.copy()

    def setup_homography_auto(self, frame_width, frame_height):
        """
        Automatic homography setup using reasonable defaults
        Assumes a typical road scene with camera mounted at medium height
        """
        global H_MATRIX
        
        print("\n" + "="*60)
        print("AUTOMATIC HOMOGRAPHY SETUP")
        print("="*60)
        print("Using automatic calibration with default road assumptions")
        print("For better accuracy, implement manual calibration with known road points")
        print("="*60 + "\n")
        
        # Define 4 points in image space (bottom portion of frame, trapezoidal)
        # These represent a rectangular area on the road
        w, h = frame_width, frame_height
        
        # Bottom trapezoid - assumes camera looking forward at road
        src_pts = np.float32([
            [w * 0.3, h * 0.6],   # top-left
            [w * 0.7, h * 0.6],   # top-right
            [w * 0.9, h * 0.95],  # bottom-right
            [w * 0.1, h * 0.95]   # bottom-left
        ])
        
        # Corresponding rectangle in world space (meters)
        # Assume this represents ~10m width x 15m depth
        road_width = 10.0
        road_depth = 15.0
        
        dst_pts = np.float32([
            [0, 0],                          # top-left
            [road_width, 0],                 # top-right
            [road_width, road_depth],        # bottom-right
            [0, road_depth]                  # bottom-left
        ])
        
        H_MATRIX = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        print("Homography matrix (automatic):")
        print(H_MATRIX)
        print("="*60 + "\n")

    def setup_homography_manual(self, frame):
        """Interactive manual calibration"""
        global H_MATRIX
        
        print("\n" + "="*60)
        print("MANUAL HOMOGRAPHY CALIBRATION")
        print("="*60)
        print("Select 4 points on the road plane")
        print("Click in order: top-left, top-right, bottom-right, bottom-left")
        print("="*60 + "\n")
        
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((x, y))
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame_copy, f"P{len(points)}", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if len(points) > 1:
                    cv2.line(frame_copy, points[-2], points[-1], (0, 255, 0), 2)
                cv2.imshow("Calibration", frame_copy)
        
        frame_copy = frame.copy()
        cv2.imshow("Calibration", frame_copy)
        cv2.setMouseCallback("Calibration", mouse_callback)
        
        print("Click 4 points... (press SPACE when done, ESC to skip)")
        while True:
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                cv2.destroyWindow("Calibration")
                return False
            if key == 32 and len(points) == 4:  # SPACE
                break
        
        cv2.destroyWindow("Calibration")
        
        if len(points) != 4:
            return False
        
        print(f"\nImage points: {points}")
        print("\nEnter real-world coordinates for each point (in meters):")
        
        world_points = []
        for i in range(4):
            while True:
                try:
                    coords = input(f"Point {i+1} (x,y in meters, e.g. '0,0'): ")
                    x, y = map(float, coords.split(','))
                    world_points.append([x, y])
                    break
                except:
                    print("Invalid format. Use: x,y")
        
        src_pts = np.array(points, dtype=np.float32)
        dst_pts = np.array(world_points, dtype=np.float32)
        
        H_MATRIX = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        print("\nHomography matrix computed!")
        print(H_MATRIX)
        print("="*60 + "\n")
        return True

    def processVideo(self, output_path="output.mp4", auto_calibrate=True):
        """Process video"""
        global FPS, H_MATRIX
        
        FPS = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        new_width = int(width * 0.5)
        new_height = int(height * 0.5)
        
        print(f"Video: {width}x{height} -> {new_width}x{new_height}")
        print(f"FPS: {FPS}, Frames: {total_frames}")
        
        # Get first frame
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Cannot read video")
            return
        
        first_frame_resized = cv2.resize(first_frame, (new_width, new_height))
        
        # Setup homography
        if auto_calibrate:
            self.setup_homography_auto(new_width, new_height)
        else:
            success = self.setup_homography_manual(first_frame_resized)
            if not success:
                print("Manual calibration failed, using automatic")
                self.setup_homography_auto(new_width, new_height)
        
        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, FPS, (new_width, new_height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (new_width, new_height))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Set frame time based on frame count and FPS
                self.frame_time = frame_count / FPS
                
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
                
                net.setInput(blob)
                layersNames = net.getLayerNames()
                outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)
                
                self.postProcess(outputs, frame, gray)
                
                out.write(frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed
                    print(f"Processed {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - {fps_actual:.1f} fps")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            cap.release()
            out.release()
            print(f"\nComplete! Output: {output_path}")
            print(f"Frames processed: {frame_count}")

