import cv2
import numpy as np

def segment_pupil(blurred_img, percentile=8):
    """
    Segment the pupil region from the preprocessed eye image.
    1. Calculate a dynamic threshold based on the specified percentile (e.g., bottom 8% dark pixels).
    2. Perform binary thresholding (inverted, so pupil is white, rest is black).
    3. Apply morphological opening (erosion then dilation) to remove small noise (lashes).
    4. Apply morphological closing (dilation then erosion) to fill specular highlights/reflections in pupil.
    """
    if blurred_img is None or blurred_img.size == 0:
        return None
        
    # 1. Calculate dynamic threshold based on percentile (pupil is the darkest region)
    thresh_val = np.percentile(blurred_img, percentile)
    
    # 2. Binary inverted thresholding
    _, thresh = cv2.threshold(blurred_img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Morphological operations
    # Define structural elements
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Open: removes noise like eyelashes, small shadows
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    
    # Close: fills in specular reflections (white spots) inside the pupil
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    
    return closed
