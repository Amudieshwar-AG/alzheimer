import os
import sys
import cv2
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

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_directories, setup_logging
from src.video_capture import EyeTrackingCapture
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.preprocessing import preprocess_eye
from src.segmentation import segment_pupil
from src.pupil_detection import detect_pupil
from src.eye_tracking import EyeTracker
from src.visualization import save_feature_fusion_plots, save_gaze_path_plot

logger = setup_logging()

def generate_3d_feature_graphs(source="demo", max_frames=150, method="mediapipe", output_dir="IMAGES", name_suffix=""):
    """
    Runs tracking on the input source and generates ONLY the 3D Feature Graphs.
    """
    setup_directories()
    os.makedirs(output_dir, exist_ok=True)
    
    capture = EyeTrackingCapture(source=source)
    is_synthetic = capture.is_synthetic
    
    face_detector = FaceDetector(method=method)
    eye_detector = EyeDetector(method=method)
    tracker = EyeTracker(buffer_size=max_frames + 50)
    
    frame_idx = 0
    print(f"\n===========================================================")
    print(f"  Generating ONLY 3D Feature Fusion Graphs")
    print(f"===========================================================")
    print(f"Source      : {source}")
    print(f"Max Frames  : {max_frames}")
    print(f"Output Path : {output_dir}")
    print(f"===========================================================")
    
    best_annotated = None
    try:
        while True:
            if max_frames and frame_idx >= max_frames:
                break
                
            success, frame = capture.read_frame()
            if not success:
                break
                
            frame_idx += 1
            
            face_bbox, face_details = face_detector.detect(frame, is_synthetic=is_synthetic)
            eye_regions = eye_detector.detect_and_crop(frame, face_bbox, face_details, is_synthetic=is_synthetic)
            
            if not eye_regions and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                crop_dict = {'img': frame.copy(), 'bbox': (0, 0, w, h)}
                eye_regions = {'left_eye': crop_dict, 'right_eye': crop_dict}
                
            if eye_regions:
                for eye_key in ['left_eye', 'right_eye']:
                    eye = eye_regions[eye_key]
                    img = eye['img']
                    if img is not None and img.size > 0:
                        _, blurred = preprocess_eye(img, blur_ksize=(5, 5))
                        segmented = segment_pupil(blurred, percentile=8)
                        mp_iris = face_details[eye_key].get('iris_center') if (face_details and face_details.get(eye_key)) else None
                        pupil = detect_pupil(img, segmented, mp_iris_center=mp_iris, eye_bbox=eye['bbox'])
                        eye['pupil'] = pupil
                        
                record = tracker.track(frame_idx, eye_regions)
                from src.visualization import draw_on_frame
                annotated = draw_on_frame(frame, eye_regions, record)
                if best_annotated is None or (30 <= frame_idx <= 120 and frame_idx % 30 == 0):
                    best_annotated = annotated.copy()
            else:
                record = tracker.track(frame_idx, {})
                
            if frame_idx % 30 == 0 or frame_idx == max_frames:
                print(f"Processed frame {frame_idx}/{max_frames}...")

    except KeyboardInterrupt:
        print("Stopped tracking.")
    finally:
        capture.release()
        face_detector.close()

    # Save face photo capture from mid-stream when looking at camera
    if best_annotated is not None:
        real_captures_dir = os.path.join(output_dir, "real_face_captures")
        os.makedirs(real_captures_dir, exist_ok=True)
        cv2.imwrite(os.path.join(real_captures_dir, "webcam_face_looking_at_camera.png"), best_annotated)

    if len(tracker.history) == 0:
        print("Error: No frames were tracked. Cannot generate 3D plots.")
        return False

    suffix = f"_{name_suffix}" if name_suffix else ""
    
    # Target file paths
    gaze_path_file = os.path.join(output_dir, f"fig7_gaze_path{suffix}.png")
    feat_fusion_file = os.path.join(output_dir, f"fig8_feature_fusion{suffix}.png")
    
    print("\nPlotting 3D Feature Fusion Ribbons & Gaze Trajectory...")
    save_gaze_path_plot(tracker.history, gaze_path_file)
    save_feature_fusion_plots(tracker.history, feat_fusion_file)
    
    # Organize outputs
    feat_3d_img = os.path.join(output_dir, f"3d_feature_fusion_graph{suffix}.png")
    feat_3d_html = os.path.join(output_dir, f"3d_feature_fusion_graph{suffix}.html")
    gaze_3d_img = os.path.join(output_dir, f"3d_gaze_path_graph{suffix}.png")
    gaze_3d_html = os.path.join(output_dir, f"3d_gaze_path_graph{suffix}.html")

    import shutil
    fig8_3d_src = os.path.join(output_dir, "fig8_feature_fusion_3d.png")
    fig8_3d_html_src = os.path.join(output_dir, "fig8_feature_fusion_3d.html")
    fig7_3d_src = os.path.join(output_dir, "fig7_gaze_path_3d.png")
    fig7_3d_html_src = os.path.join(output_dir, "fig7_gaze_path_3d.html")

    if os.path.exists(fig8_3d_src):
        shutil.move(fig8_3d_src, feat_3d_img)
    if os.path.exists(fig8_3d_html_src):
        shutil.move(fig8_3d_html_src, feat_3d_html)
    if os.path.exists(fig7_3d_src):
        shutil.move(fig7_3d_src, gaze_3d_img)
    if os.path.exists(fig7_3d_html_src):
        shutil.move(fig7_3d_html_src, gaze_3d_html)

    print("\n===========================================================")
    print(f"3D Graphs Generated Successfully in: {os.path.abspath(output_dir)}")
    print(f"1. 3D Feature Fusion Graph (PNG): {os.path.basename(feat_3d_img)}")
    print(f"2. 3D Feature Fusion Graph (Interactive HTML): {os.path.basename(feat_3d_html)}")
    print(f"3. 3D Gaze Path Graph (PNG)      : {os.path.basename(gaze_3d_img)}")
    print(f"4. 3D Gaze Path Graph (HTML)     : {os.path.basename(gaze_3d_html)}")
    print(f"===========================================================")
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate ONLY 3D Feature Fusion & Gaze Path Graphs")
    parser.add_argument('--source', type=str, default='demo', help="Input source: 'demo', 'data/input', 'webcam', or path to video file")
    parser.add_argument('--frames', type=int, default=150, help="Number of frames to process (default: 150)")
    parser.add_argument('--output', type=str, default='IMAGES', help="Output directory to save 3D graphs (default: IMAGES)")
    parser.add_argument('--suffix', type=str, default='', help="Optional suffix name for output files (e.g., 'run2')")

    args = parser.parse_args()
    generate_3d_feature_graphs(source=args.source, max_frames=args.frames, output_dir=args.output, name_suffix=args.suffix)

if __name__ == '__main__':
    main()
