import os
import sys
import cv2
import numpy as np
import argparse
import logging

# Ensure google.protobuf.runtime_version is mocked before importing mediapipe on Windows
if 'google.protobuf.runtime_version' not in sys.modules:
    try:
        from google.protobuf import runtime_version
    except ImportError:
        import types
        m = types.ModuleType('google.protobuf.runtime_version')
        m.Domain = type('Domain', (), {'PUBLIC': 1})
        m.ValidateProtobufRuntimeVersion = lambda *a, **k: None
        sys.modules['google.protobuf.runtime_version'] = m
        import google.protobuf
        google.protobuf.runtime_version = m

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_eye
from src.segmentation import segment_pupil, generate_edge_segmentation
from src.pupil_detection import detect_pupil
from src.visualization import compute_color_gradient, enhance_grayscale, draw_on_frame
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.utils import setup_logging

logger = setup_logging()

def create_grid_visualization(orig_bgr, gray_bgr, color_grad_bgr, denoised_bgr, pupil_seg_bgr, edge_seg_bgr):
    """
    Creates a 2x3 composite visualization image of all 6 features side-by-side with clean labels.
    """
    # Find standard tile dimensions based on original image
    h, w = orig_bgr.shape[:2]
    target_w = max(320, w)
    target_h = int(target_w * (h / w))
    
    def prepare_tile(img, title):
        if len(img.shape) == 2:
            tile = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            tile = img.copy()
            
        tile = cv2.resize(tile, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Add top header bar for label
        header_h = 35
        header = np.zeros((header_h, target_w, 3), dtype=np.uint8)
        header[:] = (30, 30, 30) # Dark gray background
        
        cv2.putText(header, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Border around tile
        cv2.rectangle(tile, (0, 0), (target_w - 1, target_h - 1), (60, 60, 60), 1)
        
        combined_tile = np.vstack([header, tile])
        return combined_tile

    t1 = prepare_tile(orig_bgr, "1. Original Input")
    t2 = prepare_tile(gray_bgr, "2. Grayscale")
    t3 = prepare_tile(color_grad_bgr, "3. Color Gradient")
    t4 = prepare_tile(denoised_bgr, "4. Denoised (CLAHE)")
    t5 = prepare_tile(pupil_seg_bgr, "5. Pupil Segmented")
    t6 = prepare_tile(edge_seg_bgr, "6. Edge Segmented")

    row1 = np.hstack([t1, t2, t3])
    row2 = np.hstack([t4, t5, t6])
    
    grid = np.vstack([row1, row2])
    return grid

def process_single_image(img_path, output_base_dir, face_detector=None, eye_detector=None):
    filename = os.path.basename(img_path)
    base_name, ext = os.path.splitext(filename)
    
    print(f"\nProcessing: {filename}...")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return False
        
    h, w = img.shape[:2]
    
    # 1. Detect face and eyes
    face_bbox, face_details = (None, None)
    eye_regions = None
    if face_detector and eye_detector:
        face_bbox, face_details = face_detector.detect(img)
        eye_regions = eye_detector.detect_and_crop(img, face_bbox, face_details)
        
    # If no eye regions detected, fallback to treating whole image as eye crop
    if not eye_regions:
        crop_dict = {'img': img.copy(), 'bbox': (0, 0, w, h)}
        eye_regions = {'left_eye': crop_dict, 'right_eye': crop_dict}

    # Extract target region (either detected eye crop or main image)
    target_crop = eye_regions['left_eye']['img']
    if target_crop is None or target_crop.size == 0:
        target_crop = img.copy()

    # 2. Extract Feature Maps
    # A. Grayscale
    gray_img = enhance_grayscale(target_crop)
    
    # B. Color Gradient
    color_grad = compute_color_gradient(target_crop, cv2.COLORMAP_JET)
    
    # C. Preprocessed / Denoised (CLAHE + Gaussian Blur)
    contrast_enhanced, denoised = preprocess_eye(target_crop, blur_ksize=(5, 5))
    
    # D. Pupil Segmented Mask
    pupil_segmented = segment_pupil(denoised, percentile=8)
    
    # E. Edge Segmented (High fidelity black bg with white Canny + circle outlines)
    edge_segmented = generate_edge_segmentation(target_crop)
    
    # F. Pupil detection details & Annotated overlay
    for eye_key in eye_regions:
        eye_dict = eye_regions[eye_key]
        c_img = eye_dict['img']
        if c_img is not None and c_img.size > 0:
            c_enhanced, c_denoised = preprocess_eye(c_img, blur_ksize=(5, 5))
            c_seg = segment_pupil(c_denoised, percentile=8)
            mp_iris = None
            if face_details and face_details.get(eye_key):
                mp_iris = face_details[eye_key].get('iris_center')
            pupil_res = detect_pupil(c_img, c_seg, mp_iris_center=mp_iris, eye_bbox=eye_dict['bbox'])
            eye_dict['pupil'] = pupil_res
            
    annotated = draw_on_frame(img, eye_regions, tracking_record=None)

    # 3. Create Grid View
    grid_img = create_grid_visualization(
        target_crop,
        gray_img,
        color_grad,
        denoised if denoised is not None else gray_img,
        pupil_segmented if pupil_segmented is not None else gray_img,
        edge_segmented if edge_segmented is not None else gray_img
    )

    # 4. Save Outputs to Categorized Subdirectories
    subdirs = {
        'grayscale': os.path.join(output_base_dir, 'grayscale'),
        'color_gradient': os.path.join(output_base_dir, 'color_gradient'),
        'denoised': os.path.join(output_base_dir, 'denoised'),
        'segmented': os.path.join(output_base_dir, 'segmented'),
        'edge_segmented': os.path.join(output_base_dir, 'edge_segmented'),
        'annotated': os.path.join(output_base_dir, 'annotated'),
        'grid_views': os.path.join(output_base_dir, 'grid_views'),
        'per_image': os.path.join(output_base_dir, 'per_image', base_name)
    }

    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    # Write files by feature category
    cv2.imwrite(os.path.join(subdirs['grayscale'], f"{base_name}_grayscale.jpg"), gray_img)
    cv2.imwrite(os.path.join(subdirs['color_gradient'], f"{base_name}_color_gradient.jpg"), color_grad)
    if denoised is not None:
        cv2.imwrite(os.path.join(subdirs['denoised'], f"{base_name}_denoised.jpg"), denoised)
    if pupil_segmented is not None:
        cv2.imwrite(os.path.join(subdirs['segmented'], f"{base_name}_segmented.jpg"), pupil_segmented)
    if edge_segmented is not None:
        cv2.imwrite(os.path.join(subdirs['edge_segmented'], f"{base_name}_edge_segmented.jpg"), edge_segmented)
    cv2.imwrite(os.path.join(subdirs['annotated'], f"{base_name}_annotated.jpg"), annotated)
    cv2.imwrite(os.path.join(subdirs['grid_views'], f"{base_name}_all_features_grid.jpg"), grid_img)

    # Also write a complete set into per_image subfolder for easy viewing of all outputs of a single image together
    per_img_dir = subdirs['per_image']
    cv2.imwrite(os.path.join(per_img_dir, "1_original.jpg"), target_crop)
    cv2.imwrite(os.path.join(per_img_dir, "2_grayscale.jpg"), gray_img)
    cv2.imwrite(os.path.join(per_img_dir, "3_color_gradient.jpg"), color_grad)
    if denoised is not None:
        cv2.imwrite(os.path.join(per_img_dir, "4_denoised.jpg"), denoised)
    if pupil_segmented is not None:
        cv2.imwrite(os.path.join(per_img_dir, "5_pupil_segmented.jpg"), pupil_segmented)
    if edge_segmented is not None:
        cv2.imwrite(os.path.join(per_img_dir, "6_edge_segmented.jpg"), edge_segmented)
    cv2.imwrite(os.path.join(per_img_dir, "7_annotated.jpg"), annotated)
    cv2.imwrite(os.path.join(per_img_dir, "8_all_features_grid.jpg"), grid_img)

    print(f" -> Saved all 7 feature outputs + grid view for '{filename}'")
    return True

def main():
    parser = argparse.ArgumentParser(description="Batch process images to generate Grayscale, Color Gradient, Denoised, Pupil Segmented, Edge Segmented, Annotated, and Grid view outputs.")
    parser.add_argument("--input", type=str, default="data/input", help="Input image file or directory containing images")
    parser.add_argument("--output", type=str, default="data/output/all_features", help="Output directory for generated feature maps")
    parser.add_argument("--method", type=str, default="mediapipe", choices=["mediapipe", "haar"], help="Face/Eye detection method")

    args = parser.parse_args()

    # Initialize detectors
    try:
        face_detector = FaceDetector(method=args.method)
        eye_detector = EyeDetector(method=args.method)
    except Exception as e:
        print(f"Warning: Detector initialization notice ({e}). Continuing with image crop fallback.")
        face_detector, eye_detector = None, None

    if os.path.isfile(args.input):
        process_single_image(args.input, args.output, face_detector, eye_detector)
    elif os.path.isdir(args.input):
        image_files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
        if not image_files:
            print(f"No image files found in input directory: {args.input}")
            return
        
        print(f"===========================================================")
        print(f"  Batch Processing {len(image_files)} Images for All Feature Outputs")
        print(f"===========================================================")
        print(f"Input Directory : {args.input}")
        print(f"Output Directory: {args.output}")
        print(f"Feature Types   : Grayscale, Color Gradient, Denoised, Pupil Segmented, Edge Segmented, Annotated, 6-in-1 Grid")
        print(f"===========================================================")

        success_count = 0
        for idx, fname in enumerate(image_files, 1):
            full_path = os.path.join(args.input, fname)
            print(f"[{idx}/{len(image_files)}]", end="")
            if process_single_image(full_path, args.output, face_detector, eye_detector):
                success_count += 1

        print(f"\n===========================================================")
        print(f"Batch Processing Complete: {success_count}/{len(image_files)} images processed successfully.")
        print(f"All outputs saved to folder: {os.path.abspath(args.output)}")
        print(f"===========================================================")
    else:
        print(f"Error: Input path '{args.input}' does not exist.")

    if face_detector:
        face_detector.close()

if __name__ == "__main__":
    main()
