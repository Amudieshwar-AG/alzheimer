import numpy as np
import logging

logger = logging.getLogger('EyeTracking')

class EyeTracker:
    def __init__(self, buffer_size=150, smoothing_window=3):
        self.buffer_size = buffer_size
        self.smoothing_window = smoothing_window
        
        # Track coordinate history: list of dicts {'frame': f, 'left_raw': (x,y), 'left_norm': (nx, ny), ...}
        self.history = []
        
        # Keep lists for rolling smoothing
        self.left_raw_buffer = []
        self.right_raw_buffer = []
        self.left_norm_buffer = []
        self.right_norm_buffer = []
        
    def track(self, frame_idx, eye_data):
        """
        Processes current frame eye data and updates history.
        eye_data: dict returned from eye_detection
        Each eye dict inside eye_data has: 'bbox', 'img', 'ear', 'pupil' (added by main coordinate)
        """
        left = eye_data.get('left_eye')
        right = eye_data.get('right_eye')
        
        left_raw = None
        left_norm = None
        right_raw = None
        right_norm = None
        
        # Process Left Eye (anatomical left, viewer's right)
        if left and 'pupil' in left and left['pupil'] is not None:
            bx, by, bw, bh = left['bbox']
            lx, ly = left['pupil']['center'] # local inside crop
            # Convert to global frame coordinates
            left_raw = (bx + lx, by + ly)
            # Normalize coordinates relative to bounding box (0.0 to 1.0)
            left_norm = (lx / float(bw), ly / float(bh))
            
        # Process Right Eye (anatomical right, viewer's left)
        if right and 'pupil' in right and right['pupil'] is not None:
            bx, by, bw, bh = right['bbox']
            rx, ry = right['pupil']['center']
            right_raw = (bx + rx, by + ry)
            right_norm = (rx / float(bw), ry / float(bh))
            
        # Apply smoothing
        smooth_left_raw = self._apply_smoothing(left_raw, self.left_raw_buffer)
        smooth_right_raw = self._apply_smoothing(right_raw, self.right_raw_buffer)
        smooth_left_norm = self._apply_smoothing(left_norm, self.left_norm_buffer)
        smooth_right_norm = self._apply_smoothing(right_norm, self.right_norm_buffer)
        
        record = {
            'frame': frame_idx,
            'left_raw': smooth_left_raw,
            'left_norm': smooth_left_norm,
            'right_raw': smooth_right_raw,
            'right_norm': smooth_right_norm,
            'left_ear': left.get('ear', 0.3) if left else 0.0,
            'right_ear': right.get('ear', 0.3) if right else 0.0,
            'left_pupil_radius': left['pupil']['radius'] if left and 'pupil' in left and left['pupil'] else 0.0,
            'right_pupil_radius': right['pupil']['radius'] if right and 'pupil' in right and right['pupil'] else 0.0
        }
        
        self.history.append(record)
        if len(self.history) > self.buffer_size:
            self.history.pop(0)
            
        return record
        
    def _apply_smoothing(self, new_val, buffer_list):
        if new_val is None:
            # If tracking lost, return None, but don't clear buffer immediately
            return None
            
        buffer_list.append(new_val)
        if len(buffer_list) > self.smoothing_window:
            buffer_list.pop(0)
            
        # Return average of values in buffer
        xs = [val[0] for val in buffer_list]
        ys = [val[1] for val in buffer_list]
        return (float(np.mean(xs)), float(np.mean(ys)))
        
    def get_gaze_path(self, eye='left'):
        """
        Returns list of (x,y) normalized coordinates for drawing gaze path
        """
        path = []
        key = f'{eye}_norm'
        for record in self.history:
            if record[key] is not None:
                path.append(record[key])
        return path
