import cv2
import os
import logging

logger = logging.getLogger('EyeTracking')

class EyeDetector:
    def __init__(self, method="mediapipe", cascade_path=None):
        self.method = method
        self.eye_cascade = None
        
        if method == "haar":
            if not cascade_path:
                base_dir = os.path.dirname(os.path.dirname(__file__))
                cascade_path = os.path.join(base_dir, 'models', 'haarcascade_eye.xml')
            if os.path.exists(cascade_path):
                self.eye_cascade = cv2.CascadeClassifier(cascade_path)
                if self.eye_cascade.empty():
                    logger.error(f"Error: Eye cascade at {cascade_path} is empty or failed to load. Falling back to MediaPipe.")
                    self.method = "mediapipe"
            else:
                logger.warning(f"Haar cascade XML not found at {cascade_path}. Fallback to MediaPipe.")
                self.method = "mediapipe"
                
    def detect_and_crop(self, frame, face_bbox, detection_data=None, is_synthetic=False):
        """
        Detects left and right eyes and crops them from the frame.
        Returns a dict:
        {
            'left_eye': {'bbox': (x,y,w,h), 'img': cropped_image, 'ear': ear_value, 'iris_center': (x,y)},
            'right_eye': {'bbox': (x,y,w,h), 'img': cropped_image, 'ear': ear_value, 'iris_center': (x,y)}
        }
        or None if no eyes are detected.
        """
        h, w, _ = frame.shape
        
        if is_synthetic:
            # Viewer's left eye is the anatomical right eye
            # Viewer's right eye is the anatomical left eye
            right_bbox = (200, 190, 80, 60)
            left_bbox = (360, 190, 80, 60)
            
            rx, ry, rw, rh = right_bbox
            lx, ly, lw, lh = left_bbox
            
            # Simple fallback values for EAR
            # Let's say if we detect blink, EAR drops below 0.15
            # We can read pixel value in the middle to estimate EAR, but since we are in synthetic mode,
            # we can just compute EAR from the frame pixels or pass it.
            # Let's look at the center of the crop. If it's a line, it's very dark. If open, it's white/blue.
            # A direct way: analyze the crop or compute it.
            # Since we have the actual crop, the preprocessing/segmentation will run on it!
            # Let's estimate EAR dynamically by checking if the eye is closed:
            # We can calculate the vertical height of the white area in the crop to set EAR.
            # For simplicity, we can do a quick check of the image.
            
            return {
                'left_eye': {
                    'bbox': left_bbox,
                    'img': frame[ly:ly+lh, lx:lx+lw].copy(),
                    'ear': 0.3, # Will be overridden or estimated
                    'iris_center': None
                },
                'right_eye': {
                    'bbox': right_bbox,
                    'img': frame[ry:ry+rh, rx:rx+rw].copy(),
                    'ear': 0.3,
                    'iris_center': None
                }
            }
            
        if self.method == "mediapipe":
            if not detection_data:
                return None
                
            left_data = detection_data['left_eye']
            right_data = detection_data['right_eye']
            
            lx, ly, lw, lh = left_data['bbox']
            rx, ry, rw, rh = right_data['bbox']
            
            left_img = frame[ly:ly+lh, lx:lx+lw].copy()
            right_img = frame[ry:ry+rh, rx:rx+rw].copy()
            
            return {
                'left_eye': {
                    'bbox': left_data['bbox'],
                    'img': left_img,
                    'ear': left_data['ear'],
                    'iris_center': left_data['iris_center']
                },
                'right_eye': {
                    'bbox': right_data['bbox'],
                    'img': right_img,
                    'ear': right_data['ear'],
                    'iris_center': right_data['iris_center']
                }
            }
            
        else: # Haar Cascade method
            if not face_bbox:
                return None
                
            fx, fy, fw, fh = face_bbox
            face_roi_gray = cv2.cvtColor(frame[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2GRAY)
            
            # Detect eyes within face region
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)
            if len(eyes) < 2:
                return None
                
            # Filter and sort eyes by their relative X coordinate (left to right)
            # We want the two largest eyes detected
            sorted_eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            # Now sort them viewer-left (which is anatomical right) to viewer-right (anatomical left)
            sorted_eyes = sorted(sorted_eyes, key=lambda e: e[0])
            
            right_eye_local = sorted_eyes[0]  # viewer's left = right_eye
            left_eye_local = sorted_eyes[1]   # viewer's right = left_eye
            
            # Translate back to global coordinates
            rx, ry, rw, rh = right_eye_local
            lx, ly, lw, lh = left_eye_local
            
            rx_global, ry_global = fx + rx, fy + ry
            lx_global, ly_global = fx + lx, fy + ly
            
            # Clamp to frame boundary
            rx_global, ry_global = max(0, rx_global), max(0, ry_global)
            lx_global, ly_global = max(0, lx_global), max(0, ly_global)
            
            rw = min(w - rx_global, rw)
            rh = min(h - ry_global, rh)
            lw = min(w - lx_global, lw)
            lh = min(h - ly_global, lh)
            
            right_bbox = (rx_global, ry_global, rw, rh)
            left_bbox = (lx_global, ly_global, lw, lh)
            
            right_img = frame[ry_global:ry_global+rh, rx_global:rx_global+rw].copy()
            left_img = frame[ly_global:ly_global+lh, lx_global:lx_global+lw].copy()
            
            return {
                'left_eye': {
                    'bbox': left_bbox,
                    'img': left_img,
                    'ear': 0.25, # Haar doesn't provide EAR natively, set static default
                    'iris_center': None
                },
                'right_eye': {
                    'bbox': right_bbox,
                    'img': right_img,
                    'ear': 0.25,
                    'iris_center': None
                }
            }
