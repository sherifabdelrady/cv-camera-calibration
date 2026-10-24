"""
calibvision.calibrator
======================
Checkerboard-based camera calibration using Zhang's method (cv2.calibrateCamera).

Mathematical background
-----------------------
The pinhole camera model maps a 3-D world point X = [X, Y, Z]^T to a 2-D
image point x = [u, v]^T via:

    λ [u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T

where:
    K = [[fx, 0,  cx],    — intrinsic matrix
         [0,  fy, cy],
         [0,  0,   1]]

    (fx, fy) = focal lengths in pixels
    (cx, cy) = principal point (optical centre)

Lens distortion is modelled as:
    x_distorted = x_undist * (1 + k1*r² + k2*r⁴ + k3*r⁶)
               + [2*p1*x*y + p2*(r² + 2*x²)]    ← tangential
               + [p1*(r² + 2*y²) + 2*p2*x*y]

Zhang's method uses homographies estimated from multiple checkerboard views to
recover K and D simultaneously via a closed-form initialisation followed by
non-linear refinement (Levenberg–Marquardt).

Reference: Zhang, Z. (2000). A Flexible New Technique for Camera Calibration.
           IEEE TPAMI, 22(11), 1330-1334.
           https://doi.org/10.1109/34.888718
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from calibvision.profile import CalibrationProfile

logger = logging.getLogger(__name__)


@dataclass
class CheckerboardDetection:
    """Result of a single checkerboard detection attempt."""
    image_path: Path
    image_shape: Tuple[int, int]   # (height, width)
    success: bool
    corners: Optional[np.ndarray] = None   # shape (N, 1, 2) float32


@dataclass
class CalibrationResult:
    """Full output of cv2.calibrateCamera."""
    camera_matrix: np.ndarray          # K — shape (3, 3)
    dist_coeffs: np.ndarray            # D — shape (1, 5) or (1, 8)
    rotation_vecs: List[np.ndarray]    # one per valid view
    translation_vecs: List[np.ndarray] # one per valid view
    rms_reprojection_error: float      # pixels — primary quality metric
    per_view_errors: List[float]       # reprojection error per calibration image
    valid_image_paths: List[Path]      # images that yielded a valid detection
    image_size: Tuple[int, int]        # (width, height)


class CameraCalibrator:
    """
    Estimate intrinsic camera parameters (K, D) from a set of checkerboard images.

    Usage
    -----
    >>> calibrator = CameraCalibrator(board_cols=9, board_rows=6)
    >>> detections = calibrator.detect_corners(image_paths)
    >>> result = calibrator.calibrate(detections)
    >>> profile = calibrator.to_profile(result)
    >>> profile.save("configs/saved_profiles/my_camera.yaml")
    """

    def __init__(
        self,
        board_cols: int = 9,
        board_rows: int = 6,
        square_size_mm: float = 25.0,
        subpix_window: Tuple[int, int] = (11, 11),
        subpix_max_iter: int = 30,
        subpix_epsilon: float = 0.001,
        calib_flags: int = 0,
        min_valid_frames: int = 10,
    ) -> None:
        self.board_cols = board_cols
        self.board_rows = board_rows
        self.square_size_mm = square_size_mm
        self.subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            subpix_max_iter,
            subpix_epsilon,
        )
        self.subpix_window = subpix_window
        self.calib_flags = calib_flags
        self.min_valid_frames = min_valid_frames

        # Build the object-point template: (0,0,0), (1,0,0), ..., scaled by square_size_mm
        objp = np.zeros((board_rows * board_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
        objp *= square_size_mm
        self._objp_template: np.ndarray = objp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_corners(
        self, image_paths: List[Path], verbose: bool = True
    ) -> List[CheckerboardDetection]:
        """
        Attempt checkerboard corner detection on every supplied image.

        Parameters
        ----------
        image_paths : list of Path
            Images to process. Colour or grayscale; any resolution.
        verbose : bool
            Log detection outcome per image.

        Returns
        -------
        list of CheckerboardDetection
            One entry per image. Check `.success` to filter valid detections.
        """
        detections: List[CheckerboardDetection] = []

        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("Could not read image: %s — skipping", path)
                detections.append(
                    CheckerboardDetection(path, (0, 0), success=False)
                )
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            found, corners = cv2.findChessboardCorners(
                gray, (self.board_cols, self.board_rows), None
            )

            if found:
                # Sub-pixel refinement for more accurate corner localisation
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    self.subpix_window,
                    (-1, -1),
                    self.subpix_criteria,
                )
                detections.append(
                    CheckerboardDetection(path, (h, w), success=True,
                                          corners=corners_refined)
                )
                if verbose:
                    logger.info("✓ Found corners: %s", path.name)
            else:
                detections.append(
                    CheckerboardDetection(path, (h, w), success=False)
                )
                if verbose:
                    logger.warning("✗ No corners: %s", path.name)

        n_ok = sum(d.success for d in detections)
        logger.info(
            "Corner detection complete: %d/%d images valid", n_ok, len(image_paths)
        )
        return detections

    def calibrate(
        self,
        detections: List[CheckerboardDetection],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> CalibrationResult:
        """
        Run cv2.calibrateCamera on valid detections.

        Parameters
        ----------
        detections : list of CheckerboardDetection
        image_size : (width, height) or None
            Inferred from detections if not provided.

        Returns
        -------
        CalibrationResult

        Raises
        ------
        ValueError
            If fewer than min_valid_frames valid detections are available.
        """
        valid = [d for d in detections if d.success]

        if len(valid) < self.min_valid_frames:
            raise ValueError(
                f"Only {len(valid)} valid detections — need at least "
                f"{self.min_valid_frames}. Capture more checkerboard images "
                "from different angles, distances, and positions."
            )

        # Determine image size from first valid detection
        if image_size is None:
            h, w = valid[0].image_shape
            image_size = (w, h)

        obj_points = [self._objp_template for _ in valid]
        img_points = [d.corners for d in valid]

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            image_size,
            None,
            None,
            flags=self.calib_flags,
        )

        # Compute per-view reprojection errors for diagnostic plots
        # Use numpy instead of cv2.norm — cv2.norm type-checks changed in OpenCV 5.0
        per_view_errors: List[float] = []
        for i, (d, rvec, tvec) in enumerate(zip(valid, rvecs, tvecs)):
            proj, _ = cv2.projectPoints(
                self._objp_template, rvec, tvec, camera_matrix, dist_coeffs
            )
            diff = d.corners.reshape(-1, 2) - proj.reshape(-1, 2)
            err  = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
            per_view_errors.append(err)

        logger.info(
            "Calibration complete — RMS reprojection error: %.4f px", rms
        )
        if rms > 1.0:
            logger.warning(
                "RMS error %.4f px exceeds 1.0 px. Consider: capturing more "
                "diverse checkerboard poses, checking board planarity, or "
                "using a finer sub-pixel refinement window.", rms
            )

        return CalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rotation_vecs=list(rvecs),
            translation_vecs=list(tvecs),
            rms_reprojection_error=rms,
            per_view_errors=per_view_errors,
            valid_image_paths=[d.image_path for d in valid],
            image_size=image_size,
        )

    @staticmethod
    def to_profile(
        result: CalibrationResult,
        camera_name: str = "unknown",
        notes: str = "",
    ) -> "CalibrationProfile":
        """Convert a CalibrationResult into a saveable CalibrationProfile."""
        return CalibrationProfile(
            camera_name=camera_name,
            image_size=result.image_size,
            camera_matrix=result.camera_matrix,
            dist_coeffs=result.dist_coeffs,
            rms_reprojection_error=result.rms_reprojection_error,
            notes=notes,
        )
