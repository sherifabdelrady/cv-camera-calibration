"""
calibvision.cli
===============
Command-line interface for camera calibration and image dewarping.

Usage
-----
  # Calibrate from a directory of checkerboard images
  calibvision-calibrate --images data/calibration_images --profile configs/saved_profiles/my_cam.yaml

  # Undistort a single image or a directory
  calibvision-dewarp --input path/to/image.jpg --profile configs/saved_profiles/my_cam.yaml --output output/

  # Perspective rectify (interactive four-point selection)
  calibvision-dewarp --input image.jpg --profile my_cam.yaml --rectify --output output/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from calibvision.calibrator import CameraCalibrator
from calibvision.dewarper import LensDewarper, PerspectiveRectifier
from calibvision.profile import CalibrationProfile
from calibvision.utils import load_images_from_dir, draw_reprojection_error

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# calibvision-calibrate
# ─────────────────────────────────────────────────────────────────────────────

def calibrate() -> None:
    """Entry point: estimate K, D from checkerboard images → save YAML profile."""
    parser = argparse.ArgumentParser(
        description="Camera calibration using Zhang's method (checkerboard).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images",   required=True,  help="Directory of calibration images.")
    parser.add_argument("--profile",  required=True,  help="Output YAML profile path.")
    parser.add_argument("--cols",     type=int, default=9,    help="Board inner columns.")
    parser.add_argument("--rows",     type=int, default=6,    help="Board inner rows.")
    parser.add_argument("--min",      type=int, default=10,   help="Min valid detections.")
    parser.add_argument("--alpha",    type=float, default=0.5, help="Optimal K alpha.")
    parser.add_argument("--name",     default="camera",       help="Camera name tag.")
    parser.add_argument("--verbose",  action="store_true",    help="Show per-image result.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s  %(message)s")

    img_dir = Path(args.images)
    paths, _ = load_images_from_dir(img_dir)
    if not paths:
        logger.error("No images found in %s", img_dir)
        sys.exit(1)

    cal = CameraCalibrator(
        board_cols=args.cols,
        board_rows=args.rows,
        min_valid_frames=args.min,
    )
    detections = cal.detect_corners(paths, verbose=args.verbose)
    result     = cal.calibrate(detections)
    profile    = cal.to_profile(result, camera_name=args.name)
    profile.save(args.profile)

    print(f"\n=== Calibration complete ===")
    print(profile.summary())
    print(f"\nProfile saved → {args.profile}")
    print(f"RMS reprojection error: {result.rms_reprojection_error:.4f} px")


# ─────────────────────────────────────────────────────────────────────────────
# calibvision-dewarp
# ─────────────────────────────────────────────────────────────────────────────

def dewarp() -> None:
    """Entry point: apply lens undistortion (+ optional perspective rectification)."""
    parser = argparse.ArgumentParser(
        description="Apply saved calibration profile to undistort images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",   required=True,  help="Image file or directory.")
    parser.add_argument("--profile", required=True,  help="Calibration YAML profile.")
    parser.add_argument("--output",  required=True,  help="Output directory.")
    parser.add_argument("--alpha",   type=float, default=0.5, help="Crop alpha (0–1).")
    parser.add_argument("--rectify", action="store_true",
                        help="Interactively select four corners for perspective rectification.")
    parser.add_argument("--out-w",   type=int, default=800,  help="Rectified output width.")
    parser.add_argument("--out-h",   type=int, default=600,  help="Rectified output height.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    profile  = CalibrationProfile.load(args.profile)
    dewarper = LensDewarper(profile, alpha=args.alpha)

    in_path  = Path(args.input)
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect inputs
    if in_path.is_file():
        paths = [in_path]
    elif in_path.is_dir():
        paths, _ = load_images_from_dir(in_path)
    else:
        logger.error("Input not found: %s", in_path)
        sys.exit(1)

    rectifier = PerspectiveRectifier(args.out_w, args.out_h) if args.rectify else None

    for p in paths:
        img    = cv2.imread(str(p))
        out    = dewarper.undistort(img)

        if rectifier is not None:
            pts = _pick_four_points(out, window_name=p.name)
            if pts is not None:
                out = rectifier.rectify_from_clicks(out, pts)

        dest = out_dir / p.name
        cv2.imwrite(str(dest), out)
        logger.info("Saved → %s", dest)

    print(f"\n✓ Processed {len(paths)} image(s) → {out_dir}")


def _pick_four_points(image: np.ndarray, window_name: str = "Select 4 corners") -> np.ndarray | None:
    """OpenCV mouse-click point picker — returns (4, 2) float32 array or None."""
    pts: list = []
    clone = image.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])
            cv2.circle(clone, (x, y), 6, (0, 200, 100), -1)
            cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)
    cv2.imshow(window_name, clone)
    print(f"Click 4 corners on '{window_name}' then press Enter/Space (Esc=skip)")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32) and len(pts) == 4:  # Enter or Space
            break
        if key == 27:  # Esc
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return np.array(pts, dtype=np.float32)
