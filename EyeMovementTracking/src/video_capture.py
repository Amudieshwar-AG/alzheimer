import cv2
import numpy as np
import math
import random
import logging
import os

logger = logging.getLogger('EyeTracking')

class EyeTrackingCapture:
    def __init__(self, source="webcam", width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.is_synthetic = False
        
        # Synthetic generator state
        self.frame_count = 0
        # Right eye center (viewer's left) in the 640x480 canvas
        self.right_eye_center = (240, 220)
        # Left eye center (viewer's right) in the 640x480 canvas
        self.left_eye_center = (400, 220)
        
        # Gaze targets (offsets from centers)
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.target_gaze_x = 0.0
        self.target_gaze_y = 0.0
        self.saccade_duration = 0
        self.saccade_elapsed = 0
        self.saccade_start_x = 0.0
        self.saccade_start_y = 0.0
        
        # Blink state
        self.blink_frames_remaining = 0
        self.next_blink_frame = 50
        
        # Pupil diameter modulation
        self.base_pupil_radius = 12.0
        
        if source == "webcam" or source == "0" or str(source).isdigit():
            idx = 0 if source == "webcam" else int(source)
            logger.info(f"Initializing webcam index {idx}...")
            
            # Try DirectShow API for robust Windows camera access
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(idx)
                
            if not self.cap.isOpened() and idx == 0:
                logger.info("Retrying camera index 1 with CAP_DSHOW...")
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(1)
                    
            if not self.cap.isOpened():
                logger.error(f"Failed to open webcam index {idx}. Camera permission might be blocked or camera is in use.")
                raise RuntimeError(
                    f"Could not open webcam (index {idx}). Please ensure your camera is connected, "
                    "permissions are enabled in Windows settings, and no other app (like Zoom or Teams) is using it."
                )
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.is_synthetic = False
        elif source == "demo" or source == "synthetic":
            logger.info("Initializing synthetic eye movement generator...")
            self.is_synthetic = True
        else:
            if not os.path.exists(source):
                logger.error(f"Input file '{source}' does NOT exist. Falling back to synthetic generator.")
                self.is_synthetic = True
                self.static_image = None
            else:
                ext = os.path.splitext(source)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    logger.info(f"Loading image file source: {source}...")
                    img = cv2.imread(source)
                    if img is not None:
                        self.static_image = img
                        self.is_synthetic = False
                    else:
                        logger.error(f"Could not decode image file '{source}'. Falling back to synthetic generator.")
                        self.is_synthetic = True
                else:
                    logger.info(f"Opening video file source: {source}...")
                    self.cap = cv2.VideoCapture(source)
                    self.static_image = None
                    if not self.cap.isOpened():
                        logger.error(f"Failed to open video file '{source}'. Falling back to synthetic generator.")
                        self.is_synthetic = True
                
    def read_frame(self):
        """
        Reads next frame. Returns (success, frame)
        """
        if self.is_synthetic:
            frame = self._generate_synthetic_frame()
            return True, frame
            
        if hasattr(self, 'static_image') and self.static_image is not None:
            if self.frame_count > 0:
                return False, None
            self.frame_count += 1
            frame = self.static_image.copy()
            return True, frame

        success, frame = self.cap.read()
        if not success:
            return False, None
            
        # Resize to standard size if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
            
        return True, frame
        
    def _generate_synthetic_frame(self):
        self.frame_count += 1
        
        # 1. Create dark canvas (simulates dark/low-light lab setting)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background: subtle gray pattern
        frame[:, :] = [30, 30, 30]
        
        # Draw a subtle face outline
        cv2.ellipse(frame, (320, 240), (200, 220), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(frame, (320, 240), (200, 220), 0, 0, 360, (80, 80, 80), 2)
        
        # 2. Update gaze offsets (Saccades & Fixations)
        if self.saccade_elapsed < self.saccade_duration:
            # We are in a saccade, interpolate gaze position (smooth step or linear)
            self.saccade_elapsed += 1
            t = self.saccade_elapsed / self.saccade_duration
            # Cubic interpolation for smooth eye velocity profiles
            t_smooth = t * t * (3 - 2 * t)
            self.gaze_x = self.saccade_start_x + (self.target_gaze_x - self.saccade_start_x) * t_smooth
            self.gaze_y = self.saccade_start_y + (self.target_gaze_y - self.saccade_start_y) * t_smooth
        else:
            # Fixation period - add tiny micro-saccadic drift/jitter
            self.gaze_x = self.target_gaze_x + random.uniform(-0.5, 0.5)
            self.gaze_y = self.target_gaze_y + random.uniform(-0.5, 0.5)
            
            # Saccadic trigger: every ~40-60 frames
            if self.frame_count % random.randint(40, 60) == 0:
                self.saccade_start_x = self.gaze_x
                self.saccade_start_y = self.gaze_y
                # Random target gaze offset (max 15 pixels in x, 10 pixels in y)
                self.target_gaze_x = random.uniform(-15.0, 15.0)
                self.target_gaze_y = random.uniform(-10.0, 10.0)
                self.saccade_duration = random.randint(3, 5) # saccades are very fast (3-5 frames)
                self.saccade_elapsed = 0
                
        # 3. Update Blinks
        if self.frame_count >= self.next_blink_frame:
            self.blink_frames_remaining = random.randint(3, 5) # blink duration 3-5 frames
            self.next_blink_frame = self.frame_count + random.randint(100, 150)
            
        is_blinking = False
        if self.blink_frames_remaining > 0:
            self.blink_frames_remaining -= 1
            is_blinking = True
            
        # Pupil diameter modulation over time (simulate natural pupil oscillations / hippus)
        pupil_radius = self.base_pupil_radius + 1.2 * math.sin(self.frame_count * 0.05) + random.uniform(-0.1, 0.1)
        
        # 4. Draw Eyes
        for eye_center in [self.right_eye_center, self.left_eye_center]:
            cx, cy = eye_center
            
            if is_blinking:
                # Closed eye: draw horizontal line
                cv2.line(frame, (cx - 40, cy), (cx + 40, cy), (120, 120, 120), 4)
            else:
                # Open eye:
                # Draw white sclera
                cv2.ellipse(frame, (cx, cy), (45, 25), 0, 0, 360, (240, 240, 240), -1)
                cv2.ellipse(frame, (cx, cy), (45, 25), 0, 0, 360, (150, 150, 150), 2)
                
                # Draw colored iris (blue-ish) centered at eye_center + gaze
                ix = int(cx + self.gaze_x)
                iy = int(cy + self.gaze_y)
                # Clip iris center inside sclera boundaries (approximate)
                ix = max(cx - 20, min(cx + 20, ix))
                iy = max(cy - 8, min(cy + 8, iy))
                
                cv2.circle(frame, (ix, iy), 20, (160, 100, 40), -1) # BGR: blue/brown
                cv2.circle(frame, (ix, iy), 20, (200, 140, 80), 2)
                
                # Draw dark pupil
                cv2.circle(frame, (ix, iy), int(pupil_radius), (15, 15, 15), -1)
                
                # Draw specular reflections (light reflection offsets)
                # Normal reflection (corneal reflection)
                rx1 = ix - 6
                ry1 = iy - 6
                cv2.circle(frame, (rx1, ry1), 3, (255, 255, 255), -1)
                
                # Smaller reflection
                rx2 = ix + 8
                ry2 = iy + 4
                cv2.circle(frame, (rx2, ry2), 1, (220, 220, 220), -1)
                
        # Draw eyebrows and nose for face representation
        cv2.line(frame, (210, 180), (270, 175), (80, 80, 80), 3) # Left eyebrow
        cv2.line(frame, (330, 175), (390, 180), (80, 80, 80), 3) # Right eyebrow
        cv2.line(frame, (320, 220), (320, 260), (90, 90, 90), 2) # Nose bridge
        cv2.line(frame, (310, 260), (330, 260), (90, 90, 90), 2) # Nose base
        
        # Add a bit of camera noise (Gaussian noise)
        noise = np.random.normal(0, 1.5, frame.shape).astype(np.int16)
        frame_noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame_noisy
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
            logger.info("Video capture released.")
