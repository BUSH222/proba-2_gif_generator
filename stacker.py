import argparse
import os
import numpy as np
from PIL import Image


def load_images(img_paths):
    images = []
    for p in img_paths:
        try:
            with Image.open(p) as img:
                images.append(np.array(img.convert('RGB')))
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return np.array(images)


def main():
    parser = argparse.ArgumentParser(description="Stack images in a directory.")
    parser.add_argument("path", help="Path to the input directory")
    parser.add_argument("--range", default="all", help="Image range (e.g., 'all' or '0-5')")
    parser.add_argument("--method", choices=["avg", "median", "ks"], default="avg",
                        help="Stacking method")
    parser.add_argument("-r", "--extra_rotation", type=int, default=0,
                        help="Rotate final result by steps of 90° clockwise")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.path) if f.lower().endswith(('.png', '.jpg'))]
    if not files:
        print("No images found in the specified directory.")
        return

    files.sort(key=lambda x: (x[:21], len(x), x))

    if args.range.lower() != "all":
        try:
            start_str, end_str = args.range.split('-')
            start = int(start_str)
            end = int(end_str)
            files = files[start:end+1]
        except ValueError:
            print("Invalid range format. Use 'all' or 'start-end' (e.g., '0-5').")
            return

    if not files:
        print("No images left after applying range filter.")
        return

    img_paths = [os.path.join(args.path, f) for f in files]
    print(f"Loading {len(img_paths)} images...")

    stack = load_images(img_paths)
    if len(stack) == 0:
        print("No valid images to process.")
        return

    print(f"Stacking using method: {args.method}...")

    if args.method == "avg":
        result = np.mean(stack, axis=0)
    elif args.method == "median":
        result = np.median(stack, axis=0)
    elif args.method == "ks":  # kappa-sigma klipping
        kappa = 2.0
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        mask = np.abs(stack - mean) < (kappa * std)
        masked_stack = np.ma.masked_array(stack, ~mask)
        result = np.ma.mean(masked_stack, axis=0).filled(mean)

    result = np.clip(result, 0, 255).astype(np.uint8)

    if args.extra_rotation:
        result = np.rot90(result, k=-args.extra_rotation)

    out_path = os.path.join(args.path, f"stacked_{args.method}.png")
    Image.fromarray(result).save(out_path)
    print(f"Saved stacked image to {out_path}")


if __name__ == "__main__":
    main()
