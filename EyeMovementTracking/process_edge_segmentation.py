import os
import cv2
import argparse
import sys

# Add root folder to sys.path to enable module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.segmentation import generate_edge_segmentation

def process_image(image_path, output_path, canny_low=30, canny_high=120, clahe_clip=2.0):
    print(f"Loading image: {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return False

    print(f"Processing edge segmentation (Canny: {canny_low}-{canny_high}, CLAHE clip: {clahe_clip})...")
    edge_map = generate_edge_segmentation(
        img,
        clahe_clip=clahe_clip,
        canny_low=canny_low,
        canny_high=canny_high
    )

    if edge_map is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, edge_map)
        print(f"Successfully saved edge segmentation map (Full Res: {edge_map.shape[1]}x{edge_map.shape[0]}) -> {output_path}")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Edge-based Eye Segmentation Visualization (Black Background, White Edges)")
    parser.add_argument("--input", type=str, required=True, help="Input eye crop image path or directory")
    parser.add_argument("--output", type=str, default="data/output/edge_segmented", help="Output file path or folder")
    parser.add_argument("--canny-low", type=int, default=30, help="Canny lower threshold (default: 30)")
    parser.add_argument("--canny-high", type=int, default=120, help="Canny upper threshold (default: 120)")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")

    args = parser.parse_args()

    if os.path.isfile(args.input):
        out_file = args.output if args.output.endswith(('.png', '.jpg', '.jpeg')) else os.path.join(args.output, "edge_segmented_" + os.path.basename(args.input))
        process_image(args.input, out_file, args.canny_low, args.canny_high, args.clahe_clip)
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = [f for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"No image files found in '{args.input}'")
            return
        print(f"Batch processing {len(files)} images from '{args.input}'...")
        for f in files:
            in_p = os.path.join(args.input, f)
            out_p = os.path.join(args.output, f"edge_segmented_{f}")
            process_image(in_p, out_p, args.canny_low, args.canny_high, args.clahe_clip)
    else:
        print(f"Input path '{args.input}' does not exist.")

if __name__ == "__main__":
    main()
