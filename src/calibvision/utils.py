"""
calibvision.utils
=================
Shared helpers: image I/O, visualisation, and reprojection error plotting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Supported image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_images_from_dir(
    directory: str | Path,
    extensions: Optional[set] = None,
    grayscale: bool = False,
) -> Tuple[List[Path], List[np.ndarray]]:
    """
    Load all images from a directory.

    Parameters
    ----------
    directory : str or Path
    extensions : set of str, optional
        File extensions to include. Defaults to common image formats.
    grayscale : bool
        If True, convert to single-channel grayscale.

    Returns
    -------
    (paths, images) — parallel lists
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    exts = extensions or _IMG_EXTS
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)

    if not paths:
        logger.warning("No images found in %s with extensions %s", directory, exts)
        return [], []

    images: List[np.ndarray] = []
    valid_paths: List[Path] = []

    for p in paths:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(p), flag)
        if img is None:
            logger.warning("Failed to read: %s", p)
            continue
        images.append(img)
        valid_paths.append(p)

    logger.info("Loaded %d/%d images from %s", len(images), len(paths), directory)
    return valid_paths, images


def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    board_cols: int,
    board_rows: int,
) -> np.ndarray:
    """Draw detected checkerboard corners on a copy of the image."""
    vis = image.copy()
    cv2.drawChessboardCorners(vis, (board_cols, board_rows), corners, True)
    return vis


def side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    left_label: str = "Original",
    right_label: str = "Corrected",
    label_color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 1.0,
) -> np.ndarray:
    """
    Concatenate two images horizontally with text labels.

    Resizes right image to match left's height if needed.
    """
    h = left.shape[0]
    if right.shape[0] != h:
        scale = h / right.shape[0]
        right = cv2.resize(right, None, fx=scale, fy=scale)

    # Ensure both are BGR for uniform concatenation
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

    combined = np.concatenate([left, right], axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    cv2.putText(combined, left_label,  (10, 35), font, font_scale, label_color, thickness)
    cv2.putText(combined, right_label, (left.shape[1] + 10, 35),
                font, font_scale, label_color, thickness)
    return combined


def draw_reprojection_error(
    per_view_errors: List[float],
    valid_image_names: List[str],
    rms: float,
    max_acceptable: float = 1.0,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot per-view reprojection errors as a bar chart.

    Parameters
    ----------
    per_view_errors : list of float — one value per calibration image
    valid_image_names : list of str — filenames for x-axis labels
    rms : float — overall RMS error
    max_acceptable : float — threshold line drawn on the chart
    figsize : tuple

    Returns
    -------
    matplotlib Figure (call fig.savefig() or plt.show())
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = [
        "#e74c3c" if e > max_acceptable else "#2ecc71"
        for e in per_view_errors
    ]
    bars = ax.bar(range(len(per_view_errors)), per_view_errors, color=colors,
                  edgecolor="white", linewidth=0.5)

    ax.axhline(max_acceptable, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Max acceptable ({max_acceptable:.1f} px)")
    ax.axhline(rms, color="#3498db", linestyle="-", linewidth=2,
               label=f"RMS = {rms:.4f} px")

    ax.set_xticks(range(len(per_view_errors)))
    ax.set_xticklabels(
        [n[:12] for n in valid_image_names],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Reprojection Error (pixels)")
    ax.set_title("Per-View Reprojection Errors\n"
                 "Green = within limit · Red = exceeds limit")
    ax.legend()
    ax.set_ylim(0, max(max(per_view_errors) * 1.15, max_acceptable * 1.2))
    fig.tight_layout()
    return fig


def compute_fov(camera_matrix: np.ndarray, image_size: Tuple[int, int]) -> Tuple[float, float]:
    """
    Compute horizontal and vertical field of view in degrees.

    FOV_x = 2 * arctan(width  / (2 * fx))
    FOV_y = 2 * arctan(height / (2 * fy))
    """
    w, h = image_size
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    fov_x = 2 * np.degrees(np.arctan(w / (2 * fx)))
    fov_y = 2 * np.degrees(np.arctan(h / (2 * fy)))
    return fov_x, fov_y
