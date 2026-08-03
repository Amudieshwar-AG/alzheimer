import cv2
import numpy as np

def preprocess_eye(eye_img, blur_ksize=(5, 5), use_clahe=True):
    """
    Preprocess the cropped eye image.
    1. Convert to grayscale
    2. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Smooth with Gaussian Blur to reduce high frequency noise and iris texture
    """
    if eye_img is None or eye_img.size == 0:
        return None, None
        
    # 1. Convert to gray
    if len(eye_img.shape) == 3:
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_img.copy()
        
    # 2. Local contrast enhancement (helps separate pupil from dark iris and shadows)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)
    else:
        contrast_enhanced = cv2.equalizeHist(gray)
        
    # 3. Gaussian Blur to smooth out details (reflections, eyelashes, iris patterns)
    blurred = cv2.GaussianBlur(contrast_enhanced, blur_ksize, 0)
    
    return contrast_enhanced, blurred
