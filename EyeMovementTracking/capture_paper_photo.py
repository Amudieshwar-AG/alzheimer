import os
import sys
import cv2
import numpy as np
import time
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
from src.visualization import (
    draw_on_frame,
    save_gaze_path_plot,
    save_feature_fusion_plots,
    save_grayscale_and_gradient_images
)

logger = setup_logging()

def interactive_paper_photo_capture(output_dir="IMAGES"):
    """
    Opens a live webcam preview window.
    Allows the user to pose perfectly looking directly at the camera.
    Pressing SPACEBAR captures the exact perfect frame for research paper figures.
    """
    setup_directories()
    os.makedirs(output_dir, exist_ok=True)
    real_captures_dir = os.path.join(output_dir, "real_face_captures")
    os.makedirs(real_captures_dir, exist_ok=True)

    print(f"\n===========================================================")
    print(f"  Interactive Research Paper Photo Capture Tool")
    print(f"===========================================================")
    print(f"1. Look directly at your camera lens.")
    print(f"2. Align your face comfortably in the center.")
    print(f"3. Press [SPACEBAR] to capture your PERFECT photo!")
    print(f"4. Press [Q] to quit without saving.")
    print(f"===========================================================\n")

    capture = EyeTrackingCapture(source="webcam")
    face_detector = FaceDetector(method="mediapipe")
    eye_detector = EyeDetector(method="mediapipe")
    tracker = EyeTracker(buffer_size=300)

    frame_idx = 0
    captured = False
    perfect_clean_frame = None
    perfect_annotated_frame = None
    perfect_eye_regions = None

    window_name = "Research Paper Photo Capture - Press SPACEBAR to Shoot"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    try:
        while True:
            success, frame = capture.read_frame()
            if not success:
                print("Error: Could not read frame from webcam.")
                break

            frame_idx += 1
            display_frame = frame.copy()

            face_bbox, face_details = face_detector.detect(frame)
            eye_regions = eye_detector.detect_and_crop(frame, face_bbox, face_details)

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
                annotated = draw_on_frame(frame, eye_regions, record)
            else:
                record = tracker.track(frame_idx, {})
                annotated = frame.copy()

            # Add instruction overlay bar on preview
            h, w = annotated.shape[:2]
            banner_h = 45
            cv2.rectangle(annotated, (0, h - banner_h), (w, h), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                "Look at Camera & Press [SPACEBAR] to Capture | [Q] to Exit",
                (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACEBAR
                print("\n📸 [SPACEBAR Pressed] Capturing perfect research paper photo!")
                perfect_clean_frame = frame.copy()
                perfect_annotated_frame = annotated.copy()
                perfect_eye_regions = eye_regions
                captured = True

                # Visual flash feedback on window
                flash = np.full_like(annotated, 255)
                cv2.imshow(window_name, flash)
                cv2.waitKey(100)
                break
            elif key == ord('q') or key == 27:  # Q or ESC
                print("Exited capture tool.")
                break

    finally:
        capture.release()
        face_detector.close()
        cv2.destroyAllWindows()

    if captured and perfect_annotated_frame is not None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # 1. Save to IMAGES and results folder
        p1 = os.path.join(output_dir, "perfect_paper_face_annotated.png")
        p2 = os.path.join(real_captures_dir, "perfect_paper_face_annotated.png")
        p3 = os.path.join(results_dir, "fig4_video_frame.png")

        cv2.imwrite(p1, perfect_annotated_frame)
        cv2.imwrite(p2, perfect_annotated_frame)
        cv2.imwrite(p3, perfect_annotated_frame)

        if perfect_clean_frame is not None:
            cv2.imwrite(os.path.join(output_dir, "perfect_paper_face_clean.png"), perfect_clean_frame)
            cv2.imwrite(os.path.join(real_captures_dir, "perfect_paper_face_clean.png"), perfect_clean_frame)

        # Save grayscale & color gradient face maps
        save_grayscale_and_gradient_images(perfect_eye_regions, perfect_clean_frame, base_dir)

        # Regenerate 3D graphs matching perfect capture
        save_gaze_path_plot(tracker.history, os.path.join(output_dir, "fig7_gaze_path_perfect.png"))
        save_feature_fusion_plots(tracker.history, os.path.join(output_dir, "fig8_feature_fusion_perfect.png"))

        print("\n===========================================================")
        print("🎉 PERFECT RESEARCH PAPER PHOTO CAPTURED SUCCESSFULLY!")
        print("===========================================================")
        print(f"1. Annotated Paper Photo : {os.path.abspath(p1)}")
        print(f"2. Clean Paper Photo     : {os.path.abspath(os.path.join(output_dir, 'perfect_paper_face_clean.png'))}")
        print(f"3. Figure 4 Paper Target : {os.path.abspath(p3)}")
        print(f"4. Updated 3D Plots      : Saved in {os.path.abspath(output_dir)}")
        print("===========================================================")
        return True
    else:
        print("No photo was captured.")
        return False

if __name__ == '__main__':
    interactive_paper_photo_capture()
