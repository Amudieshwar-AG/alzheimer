import argparse
import sys
import os
import types

# Ensure google.protobuf.runtime_version is mocked before importing mediapipe
if 'google.protobuf.runtime_version' not in sys.modules:
    try:
        from google.protobuf import runtime_version
    except ImportError:
        m = types.ModuleType('google.protobuf.runtime_version')
        m.Domain = type('Domain', (), {'PUBLIC': 1})
        m.ValidateProtobufRuntimeVersion = lambda *a, **k: None
        sys.modules['google.protobuf.runtime_version'] = m
        import google.protobuf
        google.protobuf.runtime_version = m

# Add root folder to sys.path to enable smooth module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Eye Movement Tracking & Cognitive Assessment System")
    parser.add_argument('--source', type=str, default='demo',
                        help="Input source: 'webcam', camera index (e.g. '0'), 'input' / path to folder, path to single image/video file, or 'demo'.")
    parser.add_argument('--index', type=int, default=0,
                        help="Index of single image to process when source is an input folder (default: 0 for 1st image).")
    parser.add_argument('--method', type=str, default='mediapipe', choices=['mediapipe', 'haar'],
                        help="Face/Eye detection method: 'mediapipe' (recommended) or 'haar' (classical cascades).")
    parser.add_argument('--max-frames', type=int, default=150,
                        help="Max frames to process. Default is 150 (approx. 5 seconds of 30FPS stream) for simulation, set to 0 for unlimited.")
    parser.add_argument('--preview', action='store_true',
                        help="Display real-time visual output window. Note: Disable this on headless servers or non-GUI terminals.")
                        
    args = parser.parse_args()
    
    # Resolve source path if source is folder or 'input'
    source_path = args.source
    if os.path.isdir(args.source) or args.source.lower() in ["input", "data/input"]:
        input_dir = "data/input" if not os.path.isdir(args.source) else args.source
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
        if not image_files:
            print(f"Error: No image files found in directory '{input_dir}'.")
            return
        
        idx = args.index % len(image_files)
        selected_file = os.path.join(input_dir, image_files[idx])
        print(f"Dataset Single-Image Mode: Selected image [{idx+1}/{len(image_files)}] -> {image_files[idx]}")
        source_path = selected_file

    # Map max-frames = 0 to None (unlimited)
    max_frames = None if args.max_frames == 0 else args.max_frames
    
    # If source is webcam but max_frames was default (150), let it run continuously unless specified
    if source_path in ['webcam', '0'] and args.max_frames == 150:
        max_frames = None
        
    print(f"===========================================================")
    print(f"  Starting Eye Movement Tracking Diagnostic System         ")
    print(f"===========================================================")
    print(f"Source: {source_path}")
    print(f"Backend Method: {args.method}")
    print(f"Max Frames: {max_frames if max_frames else 'Continuous (Press Q in window to stop)'}")
    print(f"Visual Preview: {'Enabled' if args.preview else 'Disabled (Saving results/ plots directly)'}")
    print(f"===========================================================")
    
    run_pipeline(
        source=source_path,
        method=args.method,
        max_frames=max_frames,
        show_preview=args.preview
    )

if __name__ == '__main__':
    main()
