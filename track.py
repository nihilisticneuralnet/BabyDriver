import numpy as np

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

