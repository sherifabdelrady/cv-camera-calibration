"""
calibvision._device_info
========================
Lightweight GPU/device detection utility.
Imported at package init to populate device metadata.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect() -> dict:
    """
    Detect available compute devices and return a summary dict.

    Returns
    -------
    dict with keys:
      opencv_cuda : bool
      torch_cuda  : bool
      gpu_name    : str or None
      vram_gb     : float or None
    """
    info: dict = {
        "opencv_cuda": False,
        "torch_cuda":  False,
        "gpu_name":    None,
        "vram_gb":     None,
    }

    # OpenCV CUDA
    try:
        import cv2
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            info["opencv_cuda"] = True
            logger.info("OpenCV CUDA: %d device(s) detected",
                        cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        pass

    # PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info["torch_cuda"] = True
            info["gpu_name"]   = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"]    = round(props.total_memory / 1e9, 1)
            logger.info("PyTorch CUDA: %s  %.1f GB VRAM",
                        info["gpu_name"], info["vram_gb"])
    except ImportError:
        pass

    return info


# Run once at import time — result cached as module-level constant
DEVICE_INFO = detect()
