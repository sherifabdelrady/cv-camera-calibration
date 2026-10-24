# Camera Calibration and Image Dewarping

**Author:** Sherif Abd El-Rady · Computer Vision Engineer

A production-grade Python pipeline for intrinsic camera calibration using Zhang's method, lens distortion correction, and four-point perspective rectification — built on top of OpenCV with a clean, importable Python package.

---

## Problem Statement

Every physical camera introduces two fundamentally different geometric errors:

| Error | Cause | Correction |
|---|---|---|
| **Lens distortion** | Imperfect optics (barrel/pincushion) | Intrinsic calibration — K, D |
| **Perspective warp** | Camera viewing angle | Planar homography — H |

These are mathematically independent. This project implements both with a strict separation, allowing either or both corrections to be applied in the correct order.

---

## Architecture

```
checkerboard images
        │
        ▼
 CameraCalibrator
        │  Zhang's method (cv2.calibrateCamera)
        ▼
 CalibrationProfile  ─── save/load ──→  YAML
        │
        ├──→  LensDewarper           (precomputed remap tables)
        │
        └──→  PerspectiveRectifier   (four-point homography)
```

---

## Notebook Learning Path

| # | Notebook | What you learn |
|---|---|---|
| 00 | [Problem & Setup](notebooks/00_problem_and_setup.ipynb) | Environment, smoke test |
| 01 | [Camera Geometry](notebooks/01_camera_geometry.ipynb) | Pinhole model, K matrix, distortion math |
| 02 | [Data Exploration](notebooks/02_calibration_data_exploration.ipynb) | Corner detection, image quality |
| 03 | [Intrinsic Estimation](notebooks/03_calibration_intrinsics.ipynb) | K, D estimation, profile save |
| 04 | [Lens Correction](notebooks/04_lens_distortion_correction.ipynb) | Undistortion, alpha trade-off |
| 05 | [Perspective Rect.](notebooks/05_perspective_rectification.ipynb) | Homography, accuracy measurement |
| 06 | [Quality Assessment](notebooks/06_calibration_quality_assessment.ipynb) | Reprojection error analysis |
| 07 | [Error Analysis](notebooks/07_error_analysis.ipynb) | N-image sensitivity, leave-one-out, failure modes |
| 08 | [Profile Reuse](notebooks/08_saved_profile_reuse.ipynb) | Batch inference, throughput benchmark |

---

## Setup

```bash
git clone https://github.com/sherifabdelrady/camera-calibration
cd camera-calibration
pip install -e ".[dev]"
```

---

## Calibration — Quick Start

```bash
# 1. Capture 15–30 checkerboard images from diverse angles
# 2. Run calibration
calibvision-calibrate \
  --images data/calibration_images \
  --profile configs/saved_profiles/my_camera.yaml \
  --cols 9 --rows 6

# 3. Apply to new images
calibvision-dewarp \
  --input path/to/image.jpg \
  --profile configs/saved_profiles/my_camera.yaml \
  --output output/
```

---

## Python API

```python
from calibvision import CameraCalibrator, LensDewarper, PerspectiveRectifier, CalibrationProfile
from calibvision.utils import load_images_from_dir

# — Calibrate —
paths, _ = load_images_from_dir("data/calibration_images")
cal = CameraCalibrator(board_cols=9, board_rows=6)
detections = cal.detect_corners(paths)
result = cal.calibrate(detections)
profile = cal.to_profile(result, camera_name="my_cam")
profile.save("configs/saved_profiles/my_cam.yaml")

# — Undistort —
profile  = CalibrationProfile.load("configs/saved_profiles/my_cam.yaml")
dewarper = LensDewarper(profile, alpha=0.5)
undistorted = dewarper.undistort(image)

# — Perspective rectify —
import numpy as np
rect = PerspectiveRectifier(output_width=800, output_height=600)
src_pts = np.float32([[tl], [tr], [br], [bl]])
rectified = rect.rectify(undistorted, src_pts)
```

---

## Tests

```bash
pytest tests/ -v
```

25+ tests covering calibration, profile round-trip, undistortion, homography recovery, and utilities.

---

## Evaluation

| Metric | Synthetic demo | Real camera (typical) |
|---|---|---|
| RMS reprojection error | < 0.5 px | 0.3–0.8 px |
| Perspective recovery MAE | < 5 px | Depends on image quality |
| Undistortion throughput | > 200 FPS (CPU, 960×720) | Hardware dependent |

> **Note:** Synthetic calibration images produce near-zero lens distortion by design. A real camera calibration will show measurable K and D values.

---

## Limitations

- No metric accuracy (mm measurements) without a physically measured reference target.
- Throughput numbers are CPU-only and machine-specific.
- The interactive point picker (`--rectify` flag) requires a display.
- Calibration profile is only valid at the resolution it was estimated.

---

## References

- Zhang, Z. (2000). *A Flexible New Technique for Camera Calibration*. IEEE TPAMI 22(11).  
  https://doi.org/10.1109/34.888718  
- Brown, D. C. (1966). *Decentering distortion of lenses*. Photometric Engineering.  
- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html  
- Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*. Cambridge.

---

## License

MIT — see [LICENSE](LICENSE)
