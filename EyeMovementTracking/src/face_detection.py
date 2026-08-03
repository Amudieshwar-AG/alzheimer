import cv2
import os
import logging

logger = logging.getLogger('EyeTracking')

class FaceDetector:
    def __init__(self, method="mediapipe", cascade_path=None):
        self.method = method
        self.face_cascade = None
        self.mp_wrapper = None
        
        if method == "haar":
            if not cascade_path:
                base_dir = os.path.dirname(os.path.dirname(__file__))
                cascade_path = os.path.join(base_dir, 'models', 'haarcascade_frontalface_default.xml')
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    logger.error(f"Error: Face cascade at {cascade_path} is empty or failed to load. Falling back to MediaPipe.")
                    self.method = "mediapipe"
            else:
                logger.warning(f"Haar cascade XML not found at {cascade_path}. Fallback to MediaPipe.")
                self.method = "mediapipe"
                
        if self.method == "mediapipe":
            try:
                from models.mediapipe_face_mesh import MediaPipeFaceMeshWrapper
                self.mp_wrapper = MediaPipeFaceMeshWrapper()
                logger.info("Initialized MediaPipe Face Mesh for Face Detection.")
            except Exception as e:
                logger.error(f"Failed to import/initialize MediaPipe: {e}. If using Haar, make sure XML exists.")
                # We will initialize this lazily in case dependencies are still installing.
                self.mp_wrapper = None
            
    def detect(self, frame, is_synthetic=False):
        """
        Detects face. Returns:
           face_bbox: (x, y, w, h) bounding box of the face, or None
           detection_data: dict of extra details (landmarks, eye details) or None
        """
        if is_synthetic:
            # Return static bounding box matching video_capture.py layout
            # coordinates: x_min=120, y_min=80, width=400, height=320
            return (120, 80, 400, 320), None
            
        if self.method == "mediapipe":
            if self.mp_wrapper is None:
                from models.mediapipe_face_mesh import MediaPipeFaceMeshWrapper
                self.mp_wrapper = MediaPipeFaceMeshWrapper()
            res = self.mp_wrapper.process_frame(frame)
            if res:
                return res['face_bbox'], res
            return None, None
        else: # Haar
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            if len(faces) > 0:
                # Return the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                return tuple(map(int, largest_face)), None
            return None, None
            
    def close(self):
        if self.mp_wrapper:
            self.mp_wrapper.close()
