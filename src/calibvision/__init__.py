"""
calibvision — Camera Calibration and Image Dewarping
=====================================================
Production-ready Python package for:
  - Checkerboard-based intrinsic camera calibration
  - Lens distortion correction (radial + tangential)
  - Four-point perspective rectification (planar homography)
  - Calibration profile save/load
  - Reprojection error evaluation

GPU acceleration
----------------
  - cv2.cuda.remap  (LensDewarper)  — requires OpenCV with CUDA
  - torch.cuda      (future models) — auto-detected
Both paths fall back to CPU automatically when GPU is not available.

Author: Sherif Abd El-Rady
License: MIT
"""

from calibvision.calibrator import CameraCalibrator
from calibvision.dewarper import LensDewarper, PerspectiveRectifier
from calibvision.profile import CalibrationProfile
from calibvision.utils import load_images_from_dir, draw_reprojection_error
from calibvision import _device_info

__version__ = "0.1.0"
__all__ = [
    "CameraCalibrator",
    "LensDewarper",
    "PerspectiveRectifier",
    "CalibrationProfile",
    "load_images_from_dir",
    "draw_reprojection_error",
]
