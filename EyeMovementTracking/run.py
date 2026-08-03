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
                        help="Input source: 'webcam', camera index (e.g. '0'), path to video file, or 'demo'/'synthetic' for simulation mode.")
    parser.add_argument('--method', type=str, default='mediapipe', choices=['mediapipe', 'haar'],
                        help="Face/Eye detection method: 'mediapipe' (recommended, sub-pixel iris) or 'haar' (classical cascaded filters).")
    parser.add_argument('--max-frames', type=int, default=150,
                        help="Max frames to process. Default is 150 (approx. 5 seconds of 30FPS stream) for simulation, set to 0 for unlimited.")
    parser.add_argument('--preview', action='store_true',
                        help="Display real-time visual output window. Note: Disable this on headless servers or non-GUI terminals.")
                        
    args = parser.parse_args()
    
    # Map max-frames = 0 to None (unlimited)
    max_frames = None if args.max_frames == 0 else args.max_frames
    
    # If source is webcam but max_frames was default (150), let it run continuously unless specified
    if args.source in ['webcam', '0'] and args.max_frames == 150:
        max_frames = None
        
    print(f"===========================================================")
    print(f"  Starting Eye Movement Tracking Diagnostic System         ")
    print(f"===========================================================")
    print(f"Source: {args.source}")
    print(f"Backend Method: {args.method}")
    print(f"Max Frames: {max_frames if max_frames else 'Continuous (Press Q in window to stop)'}")
    print(f"Visual Preview: {'Enabled' if args.preview else 'Disabled (Saving results/ plots directly)'}")
    print(f"===========================================================")
    
    run_pipeline(
        source=args.source,
        method=args.method,
        max_frames=max_frames,
        show_preview=args.preview
    )

if __name__ == '__main__':
    main()
