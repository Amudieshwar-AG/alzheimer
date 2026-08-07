# Eye Movement Tracking & Image Analysis

This folder implements a modular, high-fidelity eyeball tracking and cognitive assessment pipeline designed to evaluate cognitive impairments (such as early biomarkers of Alzheimer's disease).

## Features & Pipeline Workflow

1. **Video Capture** (`src/video_capture.py`): Supports real-time webcam inputs, pre-recorded video files, and a high-fidelity synthetic eye movement generator (which simulates pupil movements, fixations, saccades, and blinks) for automated testing.
2. **Face & Eye Detection** (`src/face_detection.py`, `src/eye_detection.py`, `models/mediapipe_face_mesh.py`): Integrates classic OpenCV Haar Cascades and MediaPipe Face Mesh to dynamically crop eye regions.
3. **Preprocessing** (`src/preprocessing.py`): Converts crops to grayscale and applies local contrast enhancement (CLAHE) followed by Gaussian blurring to remove noise.
4. **Segmentation** (`src/segmentation.py`): Segments the pupil using percentile-based thresholding and morphological cleanup.
5. **Pupil & Iris Tracking** (`src/pupil_detection.py`, `src/eye_tracking.py`): Computes sub-pixel pupil centroids, normalizes coordinates to handle head movement, and applies temporal smoothing filters.
6. **Feature Extraction** (`src/feature_extraction.py`): Analyzes tracking signals to compute blinks (via Eye Aspect Ratio - EAR), fixations, and saccades using the I-VT (Velocity-Threshold Identification) algorithm.
7. **Feature Fusion** (`src/feature_fusion.py`): Computes advanced clinical cognitive descriptors, such as the Gaze Stability Index, Cognitive Load Index, and a Composite Risk Score.
8. **Visualization & Report** (`src/visualization.py`, `src/utils.py`): Plots gaze paths, EAR/Pupil/Velocity charts, and compiles a comprehensive evaluation report as a PDF.

## Directory Structure

```
EyeMovementTracking/
│
├── data/
│   ├── input/                # Put video/image test sources here
│   └── output/
│       ├── segmented/        # Output of pupil segmentation masking
│       ├── denoised/         # Output of CLAHE/Gaussian filters
│       ├── grayscale/        # Output grayscale images of frames & eye crops
│       ├── color_gradient/   # Spatial intensity color gradient maps (e.g. JET/VIRIDIS)
│       ├── gaze_path/
│       ├── feature_plots/
│       └── reports/          # Stores pipeline.log and copies of reports
│
├── models/
│   ├── mediapipe_face_mesh.py
│   # Haar XML cascade files automatically downloaded here if Haar mode is selected
│   ├── haarcascade_eye.xml
│   └── haarcascade_frontalface_default.xml
│
├── src/
│   ├── __init__.py
│   ├── main.py               # Orchestrator
│   ├── video_capture.py
│   ├── face_detection.py
│   ├── eye_detection.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── pupil_detection.py
│   ├── eye_tracking.py
│   ├── feature_extraction.py
│   ├── feature_fusion.py
│   ├── visualization.py
│   └── utils.py
│
├── results/                  # Generated diagnostic figures & PDF
│   ├── fig4_video_frame.png
│   ├── fig5_segmented_eye.png
│   ├── fig6_denoised.png
│   ├── fig7_gaze_path.png
│   ├── fig8_feature_fusion.png
│   └── final_report.pdf
│
├── requirements.txt
├── README.md
└── run.py
```

## Setup & Running Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run in Demo Mode (Synthetic Simulation)
Generates high-fidelity simulated eye tracking data, processes it through the full image-processing pipeline, and writes output plots/reports:
```bash
python run.py --source demo --max-frames 150
```

### 3. Run with Webcam (Real-time Tracking)
Starts eye tracking from webcam (first camera index `0`):
```bash
python run.py --source webcam --preview
```

Press `Q` in the window to stop tracking.

### 4. Run with a Video File
```bash
python run.py --source data/input/sample_video.mp4 --preview
```
