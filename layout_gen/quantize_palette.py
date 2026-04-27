"""
Post-processing: snap every pixel in generated segmentation maps to the
nearest colour in the fixed palette (Euclidean distance in RGB space).

Usage
-----
# Process an entire directory (outputs to <input_dir>_quantized/ by default)
python quantize_palette.py --input_dir /path/to/generated/images

# Specify output directory and use 16 worker processes
python quantize_palette.py --input_dir /path/to/generated --output_dir /path/to/out --num_workers 16

# Process a single file
python quantize_palette.py --input_file /path/to/image.png --output_dir /path/to/out

# Only process files whose name contains a keyword (e.g. 'samples_cfg')
python quantize_palette.py --input_dir /path/to/generated --suffix samples_cfg

# Overwrite inputs in-place
python quantize_palette.py --input_dir /path/to/generated --inplace

# Print per-class pixel ratios (single-process only, for sanity-checking)
python quantize_palette.py --input_dir /path/to/generated --verbose --num_workers 1
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ---------------------------------------------------------------------------
# Palette definition  (module-level so worker processes inherit it via fork)
# ---------------------------------------------------------------------------
PALETTE = np.array(
    [
        [255,  0,   0  ],   # highway     -> dark red
        [255,  255,  0  ],   # building    -> dark yellow
        [34, 139, 34],   # green lands -> dark green
        [0,   0,   255],   # water       -> dark blue
        [128, 128, 128],   # ground      -> gray
    ],
    dtype=np.float32,
)

CLASS_NAMES = ["highway", "building", "green", "water", "ground"]

# ---------------------------------------------------------------------------
# Core quantisation (CPU-bound numpy; called inside worker processes)
# ---------------------------------------------------------------------------
def quantize_image(img_array: np.ndarray) -> np.ndarray:
    """
    Replace every pixel with the nearest palette colour (L2 in RGB space).

    Parameters
    ----------
    img_array : (H, W, 3) uint8 ndarray

    Returns
    -------
    (H, W, 3) uint8 ndarray – only palette colours are present
    """
    H, W, C = img_array.shape
    assert C == 3, f"Expected 3-channel RGB image, got {C} channels"

    pixels = img_array.reshape(-1, 3).astype(np.float32)          # (N, 3)

    # Broadcast: (N, 1, 3) - (1, K, 3)  →  squared distances (N, K)
    diff     = pixels[:, None, :] - PALETTE[None, :, :]            # (N, K, 3)
    sq_dists = (diff ** 2).sum(axis=2)                             # (N, K)

    nearest  = sq_dists.argmin(axis=1)                             # (N,)
    return PALETTE[nearest].astype(np.uint8).reshape(H, W, 3)


def print_class_ratio(img_array: np.ndarray) -> None:
    """Print per-class pixel percentage (sanity-check helper)."""
    H, W, _ = img_array.shape
    total = H * W
    for name, color in zip(CLASS_NAMES, PALETTE.astype(np.uint8)):
        pct = np.all(img_array == color, axis=-1).sum() / total * 100
        print(f"  {name:<12}: {pct:5.1f}%")


# ---------------------------------------------------------------------------
# Worker function  (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------
def _worker(task: tuple) -> str:
    """
    Process one (src_path_str, dst_path_str, verbose) task.
    Returns dst_path_str on success, raises on error.
    """
    src_str, dst_str, verbose = task
    src, dst = Path(src_str), Path(dst_str)

    img      = Image.open(src).convert("RGB")
    arr      = np.array(img, dtype=np.uint8)
    result   = quantize_image(arr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(dst)

    if verbose:
        print(f"\n{src.name}")
        print_class_ratio(result)

    return dst_str


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(input_dir: Path, suffix: str = "") -> list:
    return sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS and suffix in p.name
    ])


def process_one(src: Path, dst: Path, verbose: bool = False) -> None:
    """Single-file helper (used in single-file mode or num_workers=1)."""
    _worker((str(src), str(dst), verbose))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Snap generated segmentation maps to a fixed colour palette."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir",  type=str, help="Directory of images to quantize")
    group.add_argument("--input_file", type=str, help="Single image to quantize")

    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Output directory (default: <input_dir>_quantized)")
    parser.add_argument("--inplace",     action="store_true",
                        help="Overwrite source files in-place")
    parser.add_argument("--suffix",      type=str, default="",
                        help="Only process filenames containing this substring "
                             "(e.g. 'samples_cfg' to skip control/reconstruction images)")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes "
                             f"(default: all logical CPUs = {os.cpu_count()})")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print per-class ratios for each image "
                             "(forces num_workers=1 to keep output readable)")
    args = parser.parse_args()

    # verbose + parallel would interleave output – fall back to serial
    if args.verbose and args.num_workers != 1:
        print("[info] --verbose forces --num_workers 1 to keep output readable.")
        args.num_workers = 1

    # ------------------------------------------------------------------ #
    #  Single-file mode                                                    #
    # ------------------------------------------------------------------ #
    if args.input_file:
        src = Path(args.input_file)
        dst = (Path(args.output_dir) / src.name
               if args.output_dir
               else src.with_stem(src.stem + "_quantized"))
        process_one(src, dst, verbose=args.verbose)
        print(f"Saved → {dst}")
        return

    # ------------------------------------------------------------------ #
    #  Directory mode                                                      #
    # ------------------------------------------------------------------ #
    input_dir = Path(args.input_dir)
    if args.inplace:
        output_dir = input_dir
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / (input_dir.name + "_quantized")

    paths = iter_images(input_dir, suffix=args.suffix)
    if not paths:
        print(f"No images found in {input_dir}  (suffix filter: '{args.suffix}')")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(str(src), str(output_dir / src.name), args.verbose)
             for src in paths]

    workers = min(args.num_workers, len(tasks))
    print(f"Quantizing {len(tasks)} images  →  {output_dir}  "
          f"[{workers} worker{'s' if workers > 1 else ''}]")

    if workers == 1:
        # ---- serial (simpler stack traces, easier debugging) ----
        for task in tqdm(tasks, unit="img"):
            _worker(task)
    else:
        # ---- parallel via ProcessPoolExecutor -------------------
        # ProcessPoolExecutor is preferred over ThreadPoolExecutor here
        # because quantize_image is CPU-bound (numpy).  Each worker is a
        # separate Python process so the GIL is not a bottleneck.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, t): t[0] for t in tasks}
            with tqdm(total=len(futures), unit="img") as pbar:
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc:
                        print(f"\n[ERROR] {futures[fut]}: {exc}")
                    pbar.update(1)

    print("Done.")


if __name__ == "__main__":
    main()
