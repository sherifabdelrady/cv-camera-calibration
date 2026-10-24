"""
calibvision.dewarper
====================
Two distinct geometric correction operations with GPU acceleration:

1. LensDewarper   — Corrects INTRINSIC lens distortion using (K, D) from calibration.
                    This is a per-pixel mapping determined by the camera optics.
                    GPU path: cv2.cuda.remap (10-30× faster on HD/4K images)

2. PerspectiveRectifier — Corrects EXTRINSIC viewpoint distortion by computing a
                    planar homography from four user-selected points.
                    GPU path: cv2.cuda.warpPerspective

─────────────────────────────────────────────────────────────────────────────
KEY DISTINCTION (often confused):
  Lens distortion  → caused by imperfect lens geometry; corrected with K, D.
  Perspective warp → caused by camera viewing angle; corrected with homography.
  Both can appear together. Apply lens correction FIRST, then perspective.

GPU acceleration notes
  - Requires OpenCV built with CUDA support (opencv-python-headless does NOT
    include CUDA; use opencv-contrib-python or a custom CUDA build).
  - The module auto-detects GPU availability and silently falls back to CPU.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from calibvision.profile import CalibrationProfile

logger = logging.getLogger(__name__)

# Detect whether OpenCV was compiled with CUDA support
_OPENCV_CUDA = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
if _OPENCV_CUDA:
    logger.info("OpenCV CUDA detected — GPU remap/warp enabled  (%d device(s))",
                cv2.cuda.getCudaEnabledDeviceCount())
else:
    logger.debug("OpenCV CUDA not available — using CPU remap")


class LensDewarper:
    """
    Remove lens distortion from images using a saved calibration profile.

    Mathematical summary
    --------------------
    Given a distorted pixel (u_d, v_d), the undistorted pixel (u, v) is found by:

        x_d = (u_d - cx) / fx
        y_d = (v_d - cy) / fy
        r²  = x_d² + y_d²

        x_u = x_d * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x_d*y_d + p2*(r² + 2*x_d²)
        y_u = y_d * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y_d²) + 2*p2*x_d*y_d

        u = fx * x_u + cx
        v = fy * y_u + cy

    OpenCV implements this via precomputed remap tables (initUndistortRectifyMap)
    which is more efficient than per-pixel inversion.

    Usage
    -----
    >>> profile = CalibrationProfile.load("configs/saved_profiles/my_camera.yaml")
    >>> dewarper = LensDewarper(profile, alpha=0.5)
    >>> undistorted = dewarper.undistort(distorted_image)
    """

    def __init__(
        self,
        profile: CalibrationProfile,
        alpha: float = 0.5,
        output_size: Optional[Tuple[int, int]] = None,
        use_gpu: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        profile : CalibrationProfile
        alpha : float in [0, 1]
            0.0 = crop all invalid (black) pixels from the undistorted image.
            1.0 = retain all original pixels (black borders will be visible).
            0.5 = balanced: a common practical choice.
        output_size : (width, height) or None
            If None, uses the size stored in the profile.
        use_gpu : bool
            If True and OpenCV CUDA is available, use GPU-accelerated remap.
            Typical speedup: 10–30× on HD/4K images.
        """
        self.profile = profile
        self.alpha = alpha
        self._output_size = output_size or profile.image_size
        self._use_gpu = use_gpu and _OPENCV_CUDA

        K = profile.camera_matrix
        D = profile.dist_coeffs
        w, h = self._output_size

        # Compute optimal new camera matrix to control cropping
        self._new_K, self._roi = cv2.getOptimalNewCameraMatrix(
            K, D, (w, h), alpha, (w, h)
        )

        # Pre-compute pixel remap tables — done once for efficiency
        # CV_32FC1 maps are required for cv2.cuda.remap
        map_type = cv2.CV_32FC1 if self._use_gpu else cv2.CV_16SC2
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, D, None, self._new_K, (w, h), map_type
        )

        # Upload maps to GPU once
        if self._use_gpu:
            self._gpu_map1 = cv2.cuda_GpuMat()
            self._gpu_map2 = cv2.cuda_GpuMat()
            self._gpu_map1.upload(self._map1)
            self._gpu_map2.upload(self._map2)

        logger.info(
            "LensDewarper ready — %dx%d, alpha=%.2f, GPU=%s, ROI=%s",
            w, h, alpha, self._use_gpu, self._roi,
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Apply lens distortion correction to a single image.
        Uses GPU-accelerated cv2.cuda.remap when available.

        Parameters
        ----------
        image : np.ndarray — BGR or grayscale image

        Returns
        -------
        np.ndarray — undistorted image, same dtype as input
        """
        if self._use_gpu:
            return self._undistort_gpu(image)
        return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR)

    def _undistort_gpu(self, image: np.ndarray) -> np.ndarray:
        """GPU remap via cv2.cuda — 10-30× faster than CPU for HD images."""
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(image)
        gpu_dst = cv2.cuda.remap(
            gpu_src, self._gpu_map1, self._gpu_map2, cv2.INTER_LINEAR
        )
        return gpu_dst.download()

    def undistort_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply undistort to a list of images (reuses precomputed maps)."""
        return [self.undistort(img) for img in images]

    @property
    def valid_roi(self) -> Tuple[int, int, int, int]:
        """(x, y, w, h) bounding box of non-black pixels in the undistorted image."""
        return self._roi

    def crop_to_roi(self, undistorted: np.ndarray) -> np.ndarray:
        """Crop the undistorted image to the valid pixel region."""
        x, y, w, h = self._roi
        return undistorted[y : y + h, x : x + w]


class PerspectiveRectifier:
    """
    Correct perspective distortion by mapping a user-defined quadrilateral to
    a rectangle using a planar homography.

    This is SEPARATE from lens distortion correction. Perspective rectification
    changes the apparent viewpoint (e.g., makes an angled document look flat).
    It does not correct barrel or pincushion distortion.

    Mathematical summary
    --------------------
    A homography H (3×3 matrix, 8 degrees of freedom) maps points between two
    planes in projective space:

        λ [x', y', 1]^T = H [x, y, 1]^T

    With four point correspondences (src → dst), H is uniquely determined via
    Direct Linear Transform (DLT) + normalisation (cv2.getPerspectiveTransform).

    Usage
    -----
    >>> rectifier = PerspectiveRectifier(output_width=800, output_height=600)
    >>> src_pts = np.array([[tl], [tr], [br], [bl]], dtype=np.float32)
    >>> result = rectifier.rectify(image, src_pts)
    """

    def __init__(
        self,
        output_width: int = 800,
        output_height: int = 600,
    ) -> None:
        self.output_width = output_width
        self.output_height = output_height

    def rectify(
        self,
        image: np.ndarray,
        src_points: np.ndarray,
        padding: int = 0,
    ) -> np.ndarray:
        """
        Warp the quadrilateral defined by src_points to fill the output rectangle.

        Parameters
        ----------
        image : np.ndarray — source image (BGR or grayscale)
        src_points : np.ndarray, shape (4, 2) — [top-left, top-right,
                     bottom-right, bottom-left] in image pixel coordinates
        padding : int — pixels of padding to add around the output

        Returns
        -------
        np.ndarray — rectified image of shape (output_height, output_width, ...)
        """
        if src_points.shape != (4, 2):
            raise ValueError(
                f"src_points must have shape (4, 2), got {src_points.shape}. "
                "Points should be [top-left, top-right, bottom-right, bottom-left]."
            )

        w, h = self.output_width - 2 * padding, self.output_height - 2 * padding

        dst_points = np.array(
            [[padding,     padding],
             [padding + w, padding],
             [padding + w, padding + h],
             [padding,     padding + h]],
            dtype=np.float32,
        )

        H = cv2.getPerspectiveTransform(
            src_points.astype(np.float32), dst_points
        )

        rectified = cv2.warpPerspective(
            image, H,
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        logger.debug(
            "Perspective rectification complete — H condition number: %.2f",
            np.linalg.cond(H),
        )
        return rectified

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Given four (x, y) points in arbitrary order, return them as
        [top-left, top-right, bottom-right, bottom-left].

        Strategy:
          - top-left  = smallest x + y sum
          - bottom-right = largest x + y sum
          - top-right = smallest x - y difference
          - bottom-left = largest x - y difference
        """
        pts = pts.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]   # top-left
        ordered[2] = pts[np.argmax(s)]   # bottom-right
        d = np.diff(pts, axis=1).ravel()
        ordered[1] = pts[np.argmin(d)]   # top-right
        ordered[3] = pts[np.argmax(d)]   # bottom-left
        return ordered

    def rectify_from_clicks(
        self,
        image: np.ndarray,
        raw_points: np.ndarray,
        padding: int = 0,
    ) -> np.ndarray:
        """
        Like rectify(), but auto-orders four arbitrarily-ordered points first.

        Parameters
        ----------
        image : np.ndarray
        raw_points : np.ndarray, shape (4, 2) — four corner clicks in any order
        padding : int

        Returns
        -------
        np.ndarray — rectified image
        """
        ordered = self.order_points(raw_points)
        return self.rectify(image, ordered, padding=padding)
