import cv2
import numpy as np

class MediaPipeFaceMeshWrapper:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Ensure google.protobuf.runtime_version is mocked before importing mediapipe
        import sys
        import types
        if 'google.protobuf.runtime_version' not in sys.modules:
            try:
                from google.protobuf import runtime_version
            except ImportError:
                m = types.ModuleType('google.protobuf.runtime_version')
                m.Domain = type('Domain', (), {'PUBLIC': 1})
                m.ValidateProtobufRuntimeVersion = lambda *a, **k: None
                sys.modules['google.protobuf.runtime_version'] = m
                import google.protobuf
                google.protobuf.runtime_version = m
                
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Landmark indices
        # Left Eye (anatomical left, which is viewer's right)
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Right Eye (anatomical right, which is viewer's left)
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Left Iris and Right Iris
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472] # 468 is center
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477] # 473 is center
        
    def process_frame(self, frame):
        """
        Process BGR frame and return landmarks and extracted eye features
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Helper to convert landmark to pixel coordinates
        def to_pixel(landmark):
            return np.array([int(landmark.x * w), int(landmark.y * h)])
            
        left_eye_pts = np.array([to_pixel(landmarks[idx]) for idx in self.LEFT_EYE_INDICES])
        right_eye_pts = np.array([to_pixel(landmarks[idx]) for idx in self.RIGHT_EYE_INDICES])
        
        left_iris_pts = np.array([to_pixel(landmarks[idx]) for idx in self.LEFT_IRIS_INDICES])
        right_iris_pts = np.array([to_pixel(landmarks[idx]) for idx in self.RIGHT_IRIS_INDICES])
        
        # Calculate Eye Aspect Ratio (EAR)
        # Left Eye (viewer's right): vertical distances over horizontal distance
        # Indices: p386-p374, p385-p380, p362-p263
        p386 = to_pixel(landmarks[386])
        p374 = to_pixel(landmarks[374])
        p385 = to_pixel(landmarks[385])
        p380 = to_pixel(landmarks[380])
        p362 = to_pixel(landmarks[362])
        p263 = to_pixel(landmarks[263])
        
        left_ear = (np.linalg.norm(p386 - p374) + np.linalg.norm(p385 - p380)) / (2.0 * np.linalg.norm(p362 - p263) + 1e-6)
        
        # Right Eye (viewer's left): vertical distances over horizontal distance
        # Indices: p159-p145, p158-p153, p33-p133
        p159 = to_pixel(landmarks[159])
        p145 = to_pixel(landmarks[145])
        p158 = to_pixel(landmarks[158])
        p153 = to_pixel(landmarks[153])
        p33 = to_pixel(landmarks[33])
        p133 = to_pixel(landmarks[133])
        
        right_ear = (np.linalg.norm(p159 - p145) + np.linalg.norm(p158 - p153)) / (2.0 * np.linalg.norm(p33 - p133) + 1e-6)
        
        # Eye bounding boxes (padded)
        def get_bbox(pts, padding=8):
            x_min, y_min = np.min(pts, axis=0) - padding
            x_max, y_max = np.max(pts, axis=0) + padding
            # clamp
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
            
        left_bbox = get_bbox(left_eye_pts)
        right_bbox = get_bbox(right_eye_pts)
        
        # Face bounding box
        face_pts = np.array([to_pixel(landmarks[i]) for i in range(468)])
        fx, fy = np.min(face_pts, axis=0)
        fw, fh = np.max(face_pts, axis=0) - [fx, fy]
        face_bbox = (int(fx), int(fy), int(fw), int(fh))
        
        return {
            'face_bbox': face_bbox,
            'left_eye': {
                'pts': left_eye_pts,
                'bbox': left_bbox,
                'ear': left_ear,
                'iris_center': left_iris_pts[0], # center point (468)
                'iris_pts': left_iris_pts
            },
            'right_eye': {
                'pts': right_eye_pts,
                'bbox': right_bbox,
                'ear': right_ear,
                'iris_center': right_iris_pts[0], # center point (473)
                'iris_pts': right_iris_pts
            },
            'landmarks': landmarks
        }
        
    def close(self):
        self.face_mesh.close()
