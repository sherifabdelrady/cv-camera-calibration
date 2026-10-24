"""
Tests for calibvision — camera calibration and image dewarping.

Run with:  pytest tests/ -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# ── Add src/ to path ──────────────────────────────────────────────────────────
import sys
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from calibvision.calibrator import CameraCalibrator, CalibrationResult
from calibvision.dewarper import LensDewarper, PerspectiveRectifier
from calibvision.profile import CalibrationProfile
from calibvision.utils import side_by_side, compute_fov, draw_reprojection_error


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_board(cols: int = 9, rows: int = 6, sq: int = 60, border: int = 40) -> np.ndarray:
    """Flat synthetic checkerboard — no distortion, no perspective."""
    h = rows * sq + 2 * border
    w = cols * sq + 2 * border
    board = np.full((h, w), 255, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = border + r * sq, border + c * sq
                board[y0:y0+sq, x0:x0+sq] = 0
    return board


def _make_calibration_images(n: int = 15, cols: int = 9, rows: int = 6) -> list[Path]:
    """Write n synthetic checkerboard images to a temp dir and return paths."""
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(seed=0)
    paths = []
    for i in range(n):
        board = _make_board(cols, rows, sq=int(rng.integers(55, 70)))
        # Light jitter perspective so images are diverse
        jitter = 15
        h, w   = board.shape
        src    = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst    = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
        H      = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(board, H, (w, h), borderValue=200)
        p      = tmp / f"cal_{i:03d}.png"
        cv2.imwrite(str(p), warped)
        paths.append(p)
    return paths


@pytest.fixture(scope="module")
def calibration_paths():
    return _make_calibration_images(n=15)


@pytest.fixture(scope="module")
def calibration_result(calibration_paths):
    cal = CameraCalibrator(board_cols=9, board_rows=6, min_valid_frames=8)
    detections = cal.detect_corners(calibration_paths, verbose=False)
    return cal.calibrate(detections)


@pytest.fixture(scope="module")
def saved_profile(calibration_result, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("profiles")
    cal = CameraCalibrator(board_cols=9, board_rows=6, min_valid_frames=8)
    profile = cal.to_profile(calibration_result, camera_name="test_cam")
    p = tmp / "test_cam.yaml"
    profile.save(p)
    return p, profile


# ─────────────────────────────────────────────────────────────────────────────
# CameraCalibrator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCameraCalibrator:

    def test_detect_corners_finds_all_synthetic(self, calibration_paths):
        cal  = CameraCalibrator(board_cols=9, board_rows=6)
        dets = cal.detect_corners(calibration_paths, verbose=False)
        n_ok = sum(d.success for d in dets)
        # Synthetic boards should yield high detection rate
        assert n_ok >= 10, f"Expected ≥10 detections, got {n_ok}"

    def test_calibrate_returns_result(self, calibration_result):
        r = calibration_result
        assert isinstance(r, CalibrationResult)
        assert r.camera_matrix.shape == (3, 3)
        assert r.dist_coeffs.shape[1] == 5
        assert r.rms_reprojection_error >= 0

    def test_rms_below_threshold(self, calibration_result):
        # Synthetic images should calibrate cleanly
        assert calibration_result.rms_reprojection_error < 2.0, (
            f"RMS too high: {calibration_result.rms_reprojection_error:.4f} px"
        )

    def test_per_view_errors_count_matches_valid(self, calibration_result):
        r = calibration_result
        assert len(r.per_view_errors) == len(r.valid_image_paths)

    def test_calibrate_raises_on_too_few_frames(self, calibration_paths):
        cal = CameraCalibrator(board_cols=9, board_rows=6, min_valid_frames=100)
        dets = cal.detect_corners(calibration_paths[:2], verbose=False)
        with pytest.raises(ValueError, match="valid detections"):
            cal.calibrate(dets)

    def test_intrinsic_matrix_structure(self, calibration_result):
        K = calibration_result.camera_matrix
        # K must be upper triangular with K[2,2]==1 and positive focal lengths
        assert K[2, 2] == pytest.approx(1.0)
        assert K[0, 0] > 0   # fx
        assert K[1, 1] > 0   # fy
        assert K[1, 0] == pytest.approx(0.0, abs=1e-6)  # skew
        assert K[2, 0] == pytest.approx(0.0, abs=1e-6)
        assert K[2, 1] == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationProfile tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationProfile:

    def test_save_creates_yaml_file(self, saved_profile):
        path, _ = saved_profile
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_round_trip_camera_matrix(self, saved_profile):
        path, original = saved_profile
        loaded = CalibrationProfile.load(path)
        np.testing.assert_allclose(
            loaded.camera_matrix, original.camera_matrix, atol=1e-9
        )

    def test_round_trip_dist_coeffs(self, saved_profile):
        path, original = saved_profile
        loaded = CalibrationProfile.load(path)
        np.testing.assert_allclose(
            loaded.dist_coeffs, original.dist_coeffs, atol=1e-9
        )

    def test_round_trip_image_size(self, saved_profile):
        path, original = saved_profile
        loaded = CalibrationProfile.load(path)
        assert loaded.image_size == original.image_size

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CalibrationProfile.load(tmp_path / "nonexistent.yaml")

    def test_summary_contains_camera_name(self, saved_profile):
        _, profile = saved_profile
        assert "test_cam" in profile.summary()

    def test_repr(self, saved_profile):
        _, profile = saved_profile
        r = repr(profile)
        assert "test_cam" in r
        assert "rms" in r


# ─────────────────────────────────────────────────────────────────────────────
# LensDewarper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLensDewarper:

    @pytest.fixture
    def dewarper(self, saved_profile):
        _, profile = saved_profile
        return LensDewarper(profile, alpha=0.5)

    def test_undistort_shape_preserved(self, dewarper, saved_profile):
        _, profile = saved_profile
        w, h = profile.image_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        out = dewarper.undistort(dummy)
        assert out.shape == dummy.shape

    def test_undistort_batch_length(self, dewarper, saved_profile):
        _, profile = saved_profile
        w, h = profile.image_size
        imgs = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(5)]
        outs = dewarper.undistort_batch(imgs)
        assert len(outs) == 5

    def test_valid_roi_is_tuple_of_four(self, dewarper):
        roi = dewarper.valid_roi
        assert len(roi) == 4
        assert all(isinstance(v, int) for v in roi)

    def test_alpha_zero_vs_one_differ(self, saved_profile):
        _, profile = saved_profile
        w, h = profile.image_size
        img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        d0  = LensDewarper(profile, alpha=0.0)
        d1  = LensDewarper(profile, alpha=1.0)
        out0 = d0.undistort(img)
        out1 = d1.undistort(img)
        # Different alpha produces different ROI, so at least valid_roi differs
        assert d0.valid_roi != d1.valid_roi


# ─────────────────────────────────────────────────────────────────────────────
# PerspectiveRectifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPerspectiveRectifier:

    def test_rectify_output_shape(self):
        rect = PerspectiveRectifier(output_width=640, output_height=480)
        img  = np.zeros((600, 800, 3), dtype=np.uint8)
        pts  = np.float32([[50,50],[750,50],[750,550],[50,550]])
        out  = rect.rectify(img, pts)
        assert out.shape == (480, 640, 3)

    def test_rectify_known_rectangle_recovery(self):
        """Warp a flat rectangle → recover → compare pixel accuracy."""
        W, H = 400, 300
        doc  = np.full((H, W, 3), 200, dtype=np.uint8)
        cv2.rectangle(doc, (10,10), (390,290), (50, 100, 200), -1)

        canvas_w, canvas_h = 600, 500
        src = np.float32([[0,0],[W,0],[W,H],[0,H]])
        dst = np.float32([[80,60],[520,90],[540,440],[60,460]])
        Hm  = cv2.getPerspectiveTransform(src, dst)
        distorted = cv2.warpPerspective(doc, Hm, (canvas_w, canvas_h))

        rect = PerspectiveRectifier(W, H)
        recovered = rect.rectify(distorted, dst)
        mae = cv2.absdiff(
            cv2.cvtColor(doc,       cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(recovered, cv2.COLOR_BGR2GRAY)
        ).mean()
        assert mae < 20.0, f"Perspective recovery MAE too high: {mae:.2f}"

    def test_order_points_canonical(self):
        pts = np.float32([[500,400],[100,50],[50,450],[450,80]])
        ordered = PerspectiveRectifier.order_points(pts)
        # TL has smallest x+y sum
        sums = ordered.sum(axis=1)
        assert np.argmin(sums) == 0  # TL
        assert np.argmax(sums) == 2  # BR

    def test_rectify_from_clicks_matches_ordered(self):
        rect = PerspectiveRectifier(400, 300)
        img  = np.zeros((500, 600, 3), dtype=np.uint8)
        pts_ordered   = np.float32([[80,60],[520,90],[540,440],[60,460]])
        pts_arbitrary = pts_ordered[[2,0,3,1]]  # scramble
        out_ordered   = rect.rectify(img, pts_ordered)
        out_clicks    = rect.rectify_from_clicks(img, pts_arbitrary)
        diff = cv2.absdiff(out_ordered, out_clicks).max()
        assert diff < 3  # bilinear interpolation rounding only

    def test_wrong_point_count_raises(self):
        rect = PerspectiveRectifier(400, 300)
        img  = np.zeros((300, 400, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            rect.rectify(img, np.float32([[10,10],[200,10],[200,200]]))  # only 3 pts


# ─────────────────────────────────────────────────────────────────────────────
# Utility tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:

    def test_compute_fov_typical_webcam(self):
        K = np.array([[800.,0.,640.],[0.,800.,360.],[0.,0.,1.]])
        fov_x, fov_y = compute_fov(K, (1280, 720))
        assert 60.0 < fov_x < 90.0
        assert 35.0 < fov_y < 60.0

    def test_side_by_side_shape(self):
        a = np.zeros((480, 640, 3), dtype=np.uint8)
        b = np.zeros((480, 640, 3), dtype=np.uint8)
        combined = side_by_side(a, b)
        assert combined.shape == (480, 1280, 3)

    def test_draw_reprojection_error_returns_figure(self):
        import matplotlib.pyplot as plt
        errors = [0.3, 0.8, 1.2, 0.5]
        names  = [f'img_{i}' for i in range(4)]
        fig    = draw_reprojection_error(errors, names, rms=0.7)
        assert hasattr(fig, 'savefig')
        plt.close(fig)
