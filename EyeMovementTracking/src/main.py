import cv2
import os
import time
import logging
from src.utils import setup_directories, download_haar_cascades, generate_pdf_report, setup_logging
from src.video_capture import EyeTrackingCapture
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.preprocessing import preprocess_eye
from src.segmentation import segment_pupil
from src.pupil_detection import detect_pupil
from src.eye_tracking import EyeTracker
from src.feature_extraction import extract_features
from src.feature_fusion import fuse_features
from src.visualization import (
    draw_on_frame,
    save_diagnostic_crops,
    save_gaze_path_plot,
    save_feature_fusion_plots,
    save_grayscale_and_gradient_images
)

logger = setup_logging()

def run_pipeline(source="webcam", method="mediapipe", max_frames=None, show_preview=False):
    """
    Main function that runs the eye tracking and image analysis pipeline.
    """
    logger.info("Starting Eye Tracking & Image Analysis Pipeline...")
    
    # 1. Initialize folders and models
    setup_directories()
    if method == "haar":
        download_haar_cascades()
        
    base_dir = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(base_dir, 'results')
    
    # 2. Instantiate pipeline components
    capture = EyeTrackingCapture(source=source)
    is_synthetic = capture.is_synthetic
    
    face_detector = FaceDetector(method=method)
    eye_detector = EyeDetector(method=method)
    tracker = EyeTracker(buffer_size=300) # buffer size up to 10 seconds of 30 fps
    
    frame_idx = 0
    start_time = time.time()
    last_annotated = None
    last_eye_regions = None
    last_frame = None
    first_detection = True
    
    logger.info(f"Pipeline running. Input source: {source}. Method: {method}. Synthetic fallback active: {is_synthetic}")
    
    try:
        while True:
            # Check maximum frame cap (useful for automated testing)
            if max_frames and frame_idx >= max_frames:
                logger.info(f"Max frames cap ({max_frames}) reached. Exiting frames loop.")
                break
                
            success, frame = capture.read_frame()
            if not success:
                logger.info("Video source reached the end or failed to read. Exiting loop.")
                break
                
            frame_idx += 1
            last_frame = frame.copy()
            
            # A. Detect Face
            face_bbox, face_details = face_detector.detect(frame, is_synthetic=is_synthetic)
            
            # B. Detect & Crop Eyes
            eye_regions = eye_detector.detect_and_crop(frame, face_bbox, face_details, is_synthetic=is_synthetic)
            
            if not eye_regions and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                crop_dict = {'img': frame.copy(), 'bbox': (0, 0, w, h)}
                eye_regions = {'left_eye': crop_dict, 'right_eye': crop_dict}
                
            if eye_regions:
                # Process Left and Right eyes
                for eye_key in ['left_eye', 'right_eye']:
                    eye = eye_regions[eye_key]
                    img = eye['img']
                    
                    if img is not None and img.size > 0:
                        # C. Preprocessing (Grayscale + local contrast CLAHE + Gaussian Blur)
                        contrast_enhanced, blurred = preprocess_eye(img, blur_ksize=(5,5))
                        eye['denoised'] = blurred
                        
                        # D. Segmentation (dynamic percentile thresholding + morph opening/closing)
                        segmented = segment_pupil(blurred, percentile=8)
                        eye['segmented'] = segmented
                        
                        # E. Pupil Center & Aperture Detection (CV Contour fitting with MediaPipe Iris fallback)
                        mp_iris = None
                        if face_details and face_details.get(eye_key):
                            mp_iris = face_details[eye_key].get('iris_center')
                            
                        pupil = detect_pupil(img, segmented, mp_iris_center=mp_iris, eye_bbox=eye['bbox'])
                        eye['pupil'] = pupil
                        
                # F. Coordinate Tracking & Smoothing
                record = tracker.track(frame_idx, eye_regions)
                
                # G. Real-time Visualization Overlays
                annotated = draw_on_frame(frame, eye_regions, record)
                last_annotated = annotated.copy()
                
                # Store sample crops on first detection or frame 30 for instant preview
                if first_detection or frame_idx == 30:
                    save_diagnostic_crops(eye_regions, results_dir)
                    first_detection = False
                    
                last_eye_regions = eye_regions
            else:
                record = tracker.track(frame_idx, {})
                annotated = frame.copy()
                
            # Keep tracks
            if last_annotated is None:
                last_annotated = annotated.copy()
                
            # If preview enabled and GUI is available, show window
            if show_preview:
                cv2.imshow("Eye Movement Tracking & Analysis", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit ('q' key pressed).")
                    break
                    
    except KeyboardInterrupt:
        logger.info("User stopped tracking with KeyboardInterrupt (Ctrl+C).")
    except Exception as e:
        logger.error(f"Error occurred during frame processing loop: {e}", exc_info=True)
    finally:
        capture.release()
        face_detector.close()
        if show_preview:
            cv2.destroyAllWindows()
            
        # Ensure images are saved even on early exit/interrupt
        if last_eye_regions is not None or last_frame is not None:
            try:
                save_grayscale_and_gradient_images(last_eye_regions, last_frame, base_dir)
                logger.info("Saved grayscale and color gradient images for captured frames.")
            except Exception as e:
                logger.warning(f"Could not save images on cleanup: {e}")
            cv2.destroyAllWindows()
            
    # --- Post Processing & Diagnostics Generation ---
    elapsed_time = time.time() - start_time
    logger.info(f"Tracking complete. Processed {frame_idx} frames in {elapsed_time:.2f} seconds.")
    
    if len(tracker.history) == 0:
        logger.error("No frames were successfully tracked. Cannot generate report.")
        return
        
    # Save Figure 4: Video Frame annotated
    if last_annotated is not None:
        cv2.imwrite(os.path.join(results_dir, 'fig4_video_frame.png'), last_annotated)
        logger.info("Saved Figure 4 (video frame annotated).")
        
    # Extract Features
    fps = frame_idx / elapsed_time if elapsed_time > 0 else 30.0
    # clamp FPS to reasonable value in synthetic or fast execution modes
    if is_synthetic:
        fps = 30.0
        
    extracted = extract_features(tracker.history, fps=fps)
    
    # Fuse Features to generate cognitive statistics
    fused = fuse_features(extracted)
    
    # Combine features and fused results
    metrics = {**extracted, **fused}
    
    # Save Figure 7: Gaze Path Plot
    gaze_path_img = os.path.join(results_dir, 'fig7_gaze_path.png')
    save_gaze_path_plot(tracker.history, gaze_path_img)
    
    # Save Figure 8: Feature Fusion Plot
    feat_fusion_img = os.path.join(results_dir, 'fig8_feature_fusion.png')
    save_feature_fusion_plots(tracker.history, feat_fusion_img)
    
    # Save report outputs (reports folder copy and results folder copy)
    fig_paths = {
        'fig4_video_frame': os.path.join(results_dir, 'fig4_video_frame.png'),
        'fig5_segmented_eye': os.path.join(results_dir, 'fig5_segmented_eye.png'),
        'fig6_denoised': os.path.join(results_dir, 'fig6_denoised.png'),
        'fig7_gaze_path': gaze_path_img,
        'fig8_feature_fusion': feat_fusion_img
    }
    
    output_pdf = os.path.join(results_dir, 'final_report.pdf')
    generate_pdf_report(metrics, fig_paths, output_pdf)
    
    # Also save copies inside data/output/ subfolders with timestamps to preserve history
    try:
        import shutil
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        shutil.copy(output_pdf, os.path.join(base_dir, 'data', 'output', 'reports', f'final_report_{timestamp}.pdf'))
        
        fig5_src = os.path.join(results_dir, 'fig5_segmented_eye.png')
        if os.path.exists(fig5_src):
            shutil.copy(fig5_src, os.path.join(base_dir, 'data', 'output', 'segmented', f'fig5_segmented_eye_{timestamp}.png'))
            
        fig6_src = os.path.join(results_dir, 'fig6_denoised.png')
        if os.path.exists(fig6_src):
            shutil.copy(fig6_src, os.path.join(base_dir, 'data', 'output', 'denoised', f'fig6_denoised_{timestamp}.png'))
            
        shutil.copy(gaze_path_img, os.path.join(base_dir, 'data', 'output', 'gaze_path', f'fig7_gaze_path_{timestamp}.png'))
        shutil.copy(feat_fusion_img, os.path.join(base_dir, 'data', 'output', 'feature_plots', f'fig8_feature_fusion_{timestamp}.png'))
        
        # Copy 3D Supplementary Figures & Interactive HTML
        g3d = os.path.join(results_dir, 'fig7_gaze_path_3d.png')
        if os.path.exists(g3d):
            shutil.copy(g3d, os.path.join(base_dir, 'data', 'output', 'gaze_path', f'fig7_gaze_path_3d_{timestamp}.png'))
        g3d_h = os.path.join(results_dir, 'fig7_gaze_path_3d.html')
        if os.path.exists(g3d_h):
            shutil.copy(g3d_h, os.path.join(base_dir, 'data', 'output', 'gaze_path', f'fig7_gaze_path_3d_{timestamp}.html'))
            
        f3d = os.path.join(results_dir, 'fig8_feature_fusion_3d.png')
        if os.path.exists(f3d):
            shutil.copy(f3d, os.path.join(base_dir, 'data', 'output', 'feature_plots', f'fig8_feature_fusion_3d_{timestamp}.png'))
        f3d_h = os.path.join(results_dir, 'fig8_feature_fusion_3d.html')
        if os.path.exists(f3d_h):
            shutil.copy(f3d_h, os.path.join(base_dir, 'data', 'output', 'feature_plots', f'fig8_feature_fusion_3d_{timestamp}.html'))
        
        save_grayscale_and_gradient_images(last_eye_regions, last_frame, base_dir, timestamp=timestamp)
        
        logger.info(f"Successfully archived final report and curves to data/output/ with timestamp {timestamp}.")
    except Exception as e:
        logger.warning(f"Could not archive reports/plots to output subfolders: {e}")
    logger.info("=== Assessment Summary ===")
    logger.info(f"Total Frames: {metrics['total_frames']}")
    logger.info(f"Blink Count: {metrics['blink_count']}")
    logger.info(f"Fixation Count: {metrics['fixation_count']}")
    logger.info(f"Saccade Count: {metrics['saccade_count']}")
    logger.info(f"Cognitive Load Index: {metrics['cognitive_load_index']:.2f}")
    logger.info(f"Gaze Stability Index: {metrics['gaze_stability_index']:.2f}")
    logger.info(f"Composite Neurological Risk: {metrics['composite_risk_score']:.2f}")
    logger.info("==========================")
    
    return metrics
