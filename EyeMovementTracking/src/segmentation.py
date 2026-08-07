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

def generate_edge_segmentation(
    eye_crop,
    clahe_clip=2.0,
    clahe_grid=(8, 8),
    canny_low=30,
    canny_high=120,
    blur_ksize=(3, 3),
    thickness=1
):
    """
    Generates a high-fidelity edge-based segmentation visualization of an eye crop.
    
    Style:
    - Pure black background (0).
    - Pure white continuous edge contours for eyebrows, eyelashes, iris boundary, and pupil boundary (255).
    - No filled regions, no grayscale gradients.
    - Full resolution output matching the original crop.
    """
    if eye_crop is None or eye_crop.size == 0:
        return None

    # Step 1: Load crop & convert to grayscale if BGR
    if len(eye_crop.shape) == 3:
        gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = eye_crop.copy()

    h, w = gray.shape

    # Step 2: Apply CLAHE for local contrast enhancement (prevents eyelash/eyebrow edges from fading in shadows)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    # Step 3: Small 3x3 Gaussian Blur to suppress sensor noise without erasing hair-thin edges
    blurred = cv2.GaussianBlur(enhanced, blur_ksize, 0)

    # Step 3 (cont): Canny edge detection with tuned thresholds for continuous hair/lash strands
    canny_edges = cv2.Canny(blurred, canny_low, canny_high)

    # Step 4: Detect Iris outer boundary with Hough Circle Transform
    min_iris_r = int(min(w, h) * 0.15)
    max_iris_r = int(min(w, h) * 0.48)
    
    iris_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(w, h) * 0.4,
        param1=max(canny_high, 80),
        param2=25,
        minRadius=min_iris_r,
        maxRadius=max_iris_r
    )

    iris_center = None
    iris_radius = None

    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))
        best_iris = iris_circles[0][0]
        iris_center = (int(best_iris[0]), int(best_iris[1]))
        iris_radius = int(best_iris[2])

    # Step 5: Detect Pupil boundary (smaller radius range, using iris center as prior if available)
    min_pupil_r = int(min(w, h) * 0.05)
    max_pupil_r = int(min(w, h) * 0.25)

    pupil_center = None
    pupil_radius = None

    if iris_center is not None:
        cx, cy = iris_center
        roi_pad = max_pupil_r + 10
        x1, y1 = max(0, cx - roi_pad), max(0, cy - roi_pad)
        x2, y2 = min(w, cx + roi_pad), min(h, cy + roi_pad)
        pupil_roi = blurred[y1:y2, x1:x2]

        if pupil_roi.size > 0:
            pupil_circles = cv2.HoughCircles(
                pupil_roi,
                cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=10,
                param1=max(canny_high, 80),
                param2=18,
                minRadius=min_pupil_r,
                maxRadius=max_pupil_r
            )
            if pupil_circles is not None:
                pupil_circles = np.uint16(np.around(pupil_circles))
                best_pupil = pupil_circles[0][0]
                pupil_center = (int(x1 + best_pupil[0]), int(y1 + best_pupil[1]))
                pupil_radius = int(best_pupil[2])

    if pupil_center is None:
        pupil_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(w, h) * 0.3,
            param1=max(canny_high, 80),
            param2=20,
            minRadius=min_pupil_r,
            maxRadius=max_pupil_r
        )
        if pupil_circles is not None:
            pupil_circles = np.uint16(np.around(pupil_circles))
            best_pupil = pupil_circles[0][0]
            pupil_center = (int(best_pupil[0]), int(best_pupil[1]))
            pupil_radius = int(best_pupil[2])

    # Step 6: Composite onto all-black canvas at full original resolution
    canvas = np.zeros((h, w), dtype=np.uint8)

    # (a) Canny edge map (eyebrow/eyelash strands)
    canvas = cv2.bitwise_or(canvas, canny_edges)

    # (b) Draw Iris circle outline in white (no fill)
    if iris_center is not None and iris_radius is not None:
        cv2.circle(canvas, iris_center, iris_radius, 255, thickness=thickness, lineType=cv2.LINE_8)

    # (c) Draw Pupil circle outline in white (no fill)
    if pupil_center is not None and pupil_radius is not None:
        cv2.circle(canvas, pupil_center, pupil_radius, 255, thickness=thickness, lineType=cv2.LINE_8)

    return canvas
