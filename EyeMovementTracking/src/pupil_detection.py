import cv2
import numpy as np
import math

def detect_pupil(cropped_eye, segmented_eye, mp_iris_center=None, eye_bbox=None):
    """
    Detect the pupil center and size in the cropped eye image.
    
    Returns a dict:
    {
        'center': (cx, cy),          # local coordinates inside crop
        'radius': radius,
        'ellipse': ellipse_obj,      # ((cx, cy), (w, h), angle) or None
        'contour': contour,
        'method': 'contour' or 'mediapipe' or 'fail'
    }
    or None if detection fails completely.
    """
    if cropped_eye is None or segmented_eye is None or segmented_eye.size == 0:
        return None
        
    h, w = segmented_eye.shape
    crop_center = np.array([w / 2.0, h / 2.0])
    
    # 1. Find external contours
    contours, _ = cv2.findContours(segmented_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour = None
    best_score = -1.0
    best_ellipse = None
    best_circle_data = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > (w * h * 0.5): # filter very small or very large areas
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < 0.3: # filter highly irregular shapes
            continue
            
        # Fit ellipse or circle to find center
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (axes_w, axes_h), angle = ellipse
                # check aspect ratio
                if axes_h == 0: continue
                aspect_ratio = axes_w / axes_h
                if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                    continue
                radius = (axes_w + axes_h) / 4.0
            except:
                ellipse = None
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        else:
            ellipse = None
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            
        # Distance from the center of the crop (pupil is usually in the center of the crop)
        dist = np.linalg.norm(np.array([cx, cy]) - crop_center)
        
        # Scoring function: prefer larger, more circular contours that are closer to the center
        # We normalize distance using crop size
        norm_dist = dist / (math.sqrt(w**2 + h**2) + 1e-6)
        score = circularity * area / (1.0 + 3.0 * norm_dist)
        
        if score > best_score:
            best_score = score
            best_contour = cnt
            best_ellipse = ellipse
            best_circle_data = ((int(cx), int(cy)), int(radius))
            
    if best_contour is not None:
        cx, cy = best_circle_data[0]
        radius = best_circle_data[1]
        
        return {
            'center': (cx, cy),
            'radius': radius,
            'ellipse': best_ellipse,
            'contour': best_contour,
            'method': 'contour'
        }
        
    # 2. Fallback to MediaPipe Iris landmark center if contour detection failed
    if mp_iris_center is not None and eye_bbox is not None:
        bx, by, bw, bh = eye_bbox
        # Translate global MediaPipe iris coordinates to local eye crop coordinates
        gx, gy = mp_iris_center
        local_cx = int(gx - bx)
        local_cy = int(gy - by)
        
        # Verify if inside the crop
        if 0 <= local_cx < w and 0 <= local_cy < h:
            # Estimate radius based on box height (typically iris is ~30-40% of box height)
            est_radius = int(bh * 0.18) # Pupil is smaller than full iris
            return {
                'center': (local_cx, local_cy),
                'radius': est_radius,
                'ellipse': None,
                'contour': None,
                'method': 'mediapipe'
            }
            
    return {
        'center': (int(w/2), int(h/2)),
        'radius': int(h*0.15),
        'ellipse': None,
        'contour': None,
        'method': 'fail'
    }
