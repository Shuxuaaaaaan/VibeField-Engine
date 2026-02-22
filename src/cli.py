import argparse
import sys
from engine import process_video

def main():
    parser = argparse.ArgumentParser(description="VibeField-Engine: Extract physical vibration vector fields from EVM/PVM amplified videos.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file or directory.")
    parser.add_argument("-p", "--patch-size", type=int, default=1, help="Size of the feature block N x N for spatial aggregation.")
    parser.add_argument("-f", "--freq-range", type=str, default="1,50", help="The frequency range of interest, comma-separated e.g. 10,50 (Hz).")
    parser.add_argument("-s", "--saliency-threshold", type=float, default=1e-5, help="Threshold for saliency filtering. Regions with subpixel variance below this threshold will be skipped.")
    parser.add_argument("-o", "--output-dir", type=str, default="data/results", help="Output directory for the .zarr files.")

    args = parser.parse_args()

    try:
        freq_min, freq_max = map(float, args.freq_range.split(","))
    except ValueError:
        print("Error: --freq-range must be strictly comma-separated floats like '10,50'", file=sys.stderr)
        sys.exit(1)

    print(f"Starting VibeField-Engine")
    print(f"Input: {args.input}")
    print(f"Patch Size: {args.patch_size}")
    print(f"Freq Range: [{freq_min}, {freq_max}] Hz")
    print(f"Saliency Threshold: {args.saliency_threshold}")
    print(f"Output Dir: {args.output_dir}")

    # Process the video
    try:
        process_video(
            input_path=args.input,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            freq_min=freq_min,
            freq_max=freq_max,
            saliency_threshold=args.saliency_threshold
        )
    except Exception as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
