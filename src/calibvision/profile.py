"""
calibvision.profile
===================
Serialisable calibration profile — saves/loads (K, D) to/from YAML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CalibrationProfile:
    """
    A portable representation of a camera's intrinsic calibration.

    Attributes
    ----------
    camera_name : str
        Human-readable label for the camera (e.g. "logitech_c920_1080p").
    image_size : (width, height)
        The resolution at which calibration was performed.
    camera_matrix : np.ndarray, shape (3, 3)
        Intrinsic matrix K.
    dist_coeffs : np.ndarray, shape (1, 5) or (1, 8)
        Distortion coefficients [k1, k2, p1, p2, k3, ...].
    rms_reprojection_error : float
        RMS reprojection error from calibration (pixels).
    notes : str
        Optional free-form metadata (date, lens, focal length, etc.).
    """

    camera_name: str
    image_size: Tuple[int, int]         # (width, height)
    camera_matrix: np.ndarray           # (3, 3)
    dist_coeffs: np.ndarray             # (1, N)
    rms_reprojection_error: float = 0.0
    notes: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the calibration profile to a YAML file.

        Only portable numeric data is saved — no binary blobs.
        The output file can be read without this package using standard YAML.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "camera_name": self.camera_name,
            "image_size": list(self.image_size),
            "rms_reprojection_error": float(self.rms_reprojection_error),
            "notes": self.notes,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Calibration profile saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        """
        Load a calibration profile from a YAML file.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        CalibrationProfile

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        KeyError
            If required fields are missing from the YAML.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Calibration profile not found: {path}\n"
                "Run calibration first to generate a profile."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        profile = cls(
            camera_name=data["camera_name"],
            image_size=tuple(data["image_size"]),
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
            rms_reprojection_error=float(data.get("rms_reprojection_error", 0.0)),
            notes=data.get("notes", ""),
        )

        logger.info(
            "Loaded calibration profile '%s' from %s  [RMS=%.4f px]",
            profile.camera_name,
            path,
            profile.rms_reprojection_error,
        )
        return profile

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the profile."""
        K = self.camera_matrix
        D = self.dist_coeffs.ravel()
        lines = [
            f"Camera: {self.camera_name}",
            f"Image size: {self.image_size[0]}x{self.image_size[1]} px",
            f"RMS reprojection error: {self.rms_reprojection_error:.4f} px",
            f"fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}",
            f"Distortion coefficients: k1={D[0]:.6f}  k2={D[1]:.6f}  "
            f"p1={D[2]:.6f}  p2={D[3]:.6f}  k3={D[4]:.6f}",
        ]
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CalibrationProfile(camera='{self.camera_name}', "
            f"size={self.image_size}, rms={self.rms_reprojection_error:.4f} px)"
        )
