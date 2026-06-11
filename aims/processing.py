"""
AIMS Research Console – aims/processing.py
Precision-upgraded aggregate image analysis pipeline.

Improvements over baseline:
  1. Preprocessing  – CLAHE equalisation + bilateral denoising
  2. Calibration    – checkerboard distortion removal + mm/px conversion
  3. Segmentation   – Otsu multi-level + marker-based watershed
  4. Measurement    – sub-pixel contour moments, Feret diameter,
                      angularity index (corner-sharpness), elongation
  5. Surface texture – GLCM (contrast, correlation, energy, homogeneity)
                       + Daubechies wavelet energy decomposition
  6. Validation     – per-particle confidence score + outlier flag
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

# Optional heavy dependencies – degrade gracefully if absent
try:
    from skimage.feature import grayscale_features  # type: ignore
    _HAS_SKIMAGE_GLCM = False          # we use our own GLCM below
except ImportError:
    _HAS_SKIMAGE_GLCM = False

try:
    import pywt                        # PyWavelets
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False
    warnings.warn("PyWavelets not installed – wavelet texture disabled.")

# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    mm_per_px: float                  # mean scale factor
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs:   Optional[np.ndarray] = None
    reprojection_error: float = 0.0
    used_checkerboard: bool = False


@dataclass
class ParticleMetrics:
    particle_id: int

    # Size
    area_mm2:          float = 0.0
    perimeter_mm:      float = 0.0
    equiv_diameter_mm: float = 0.0   # from area
    feret_max_mm:      float = 0.0   # caliper max
    feret_min_mm:      float = 0.0   # caliper min

    # Shape
    elongation:        float = 0.0   # feret_max / feret_min
    circularity:       float = 0.0   # 4π·A / P²
    convexity:         float = 0.0   # convex-hull area / particle area
    angularity_index:  float = 0.0   # normalised corner sharpness (0–1)
    solidity:          float = 0.0   # area / convex-hull area

    # Texture (GLCM)
    glcm_contrast:     float = 0.0
    glcm_correlation:  float = 0.0
    glcm_energy:       float = 0.0
    glcm_homogeneity:  float = 0.0

    # Texture (wavelet) – only if pywt available
    wavelet_energy_approx: float = 0.0
    wavelet_energy_detail: float = 0.0

    # Quality
    confidence:        float = 1.0   # 0–1; <0.6 = unreliable
    outlier_flag:      bool  = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    particles: List[ParticleMetrics] = field(default_factory=list)
    calibration: Optional[CalibrationResult] = None
    total_particles: int = 0
    rejected_particles: int = 0
    image_shape: Tuple[int, int] = (0, 0)

    def summary_stats(self) -> Dict[str, Any]:
        vals = {
            k: [getattr(p, k) for p in self.particles if not p.outlier_flag]
            for k in ("area_mm2", "feret_max_mm", "elongation",
                      "circularity", "angularity_index", "glcm_contrast")
        }
        out: Dict[str, Any] = {}
        for k, v in vals.items():
            if v:
                out[k] = {
                    "mean":   float(np.mean(v)),
                    "std":    float(np.std(v)),
                    "min":    float(np.min(v)),
                    "max":    float(np.max(v)),
                    "median": float(np.median(v)),
                }
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 1. Calibration
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_from_checkerboard(
    calib_images: List[np.ndarray],
    grid_shape: Tuple[int, int] = (9, 6),
    square_size_mm: float = 5.0,
) -> CalibrationResult:
    """
    Estimate camera intrinsics + distortion from checkerboard images.
    Falls back to identity if corner detection fails.
    """
    obj_pts: List[np.ndarray] = []
    img_pts: List[np.ndarray] = []

    objp = np.zeros((grid_shape[0] * grid_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_shape[0], 0:grid_shape[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    h, w = calib_images[0].shape[:2]

    for img in calib_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        ret, corners = cv2.findChessboardCorners(gray, grid_shape, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners_refined)

    if len(obj_pts) < 3:
        warnings.warn("Checkerboard calibration failed – using fallback scalar.")
        return CalibrationResult(mm_per_px=0.1, used_checkerboard=False)

    ret, cam_mat, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, (w, h), None, None
    )

    # mm/px = square_size_mm / mean-corner-spacing-in-px
    spacings = []
    for corners in img_pts:
        pts = corners.reshape(-1, 2)
        # horizontal neighbours
        dx = np.diff(pts[:grid_shape[0], 0], axis=0)
        spacings.extend(np.abs(dx).tolist())
    mm_per_px = square_size_mm / float(np.mean(spacings)) if spacings else 0.1

    return CalibrationResult(
        mm_per_px=mm_per_px,
        camera_matrix=cam_mat,
        dist_coeffs=dist,
        reprojection_error=float(ret),
        used_checkerboard=True,
    )


def calibrate_scalar(mm_per_px: float) -> CalibrationResult:
    """Simple single-value calibration (legacy / quick mode)."""
    return CalibrationResult(mm_per_px=mm_per_px)


def undistort_image(
    image: np.ndarray,
    calib: CalibrationResult,
) -> np.ndarray:
    if calib.camera_matrix is None or calib.dist_coeffs is None:
        return image
    h, w = image.shape[:2]
    new_cam, roi = cv2.getOptimalNewCameraMatrix(
        calib.camera_matrix, calib.dist_coeffs, (w, h), 1, (w, h)
    )
    dst = cv2.undistort(image, calib.camera_matrix, calib.dist_coeffs, None, new_cam)
    x, y, rw, rh = roi
    return dst[y:y + rh, x:x + rw] if rw > 0 and rh > 0 else dst


# ──────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(
    image: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_tile: Tuple[int, int] = (8, 8),
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Convert to greyscale → CLAHE contrast enhancement → bilateral denoising.
    Returns uint8 greyscale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    enhanced = clahe.apply(gray)

    # Bilateral filter (edge-preserving denoising)
    denoised = cv2.bilateralFilter(
        enhanced, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
    )
    return denoised


# ──────────────────────────────────────────────────────────────────────────────
# 3. Segmentation
# ──────────────────────────────────────────────────────────────────────────────

def segment_watershed(
    gray: np.ndarray,
    min_area_px: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-level Otsu thresholding + distance-transform watershed.
    Returns (binary_mask, label_map).
    """
    # Multi-level Otsu (3 classes → pick the brightest threshold)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background / foreground via distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    # Watershed requires colour image
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)
    markers[markers == -1] = 0   # boundary → background

    # Filter small regions
    for label in range(2, n_markers + 1):
        if np.sum(markers == label) < min_area_px:
            markers[markers == label] = 0

    binary = (markers > 1).astype(np.uint8) * 255
    return binary, markers


def segment_simple(
    gray: np.ndarray,
    min_area_px: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback Otsu without watershed (faster, less accurate)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    n, labels = cv2.connectedComponents(cleaned)
    for lbl in range(1, n):
        if np.sum(labels == lbl) < min_area_px:
            labels[labels == lbl] = 0
    binary_out = (labels > 0).astype(np.uint8) * 255
    return binary_out, labels


# ──────────────────────────────────────────────────────────────────────────────
# 4. Feret diameter (rotating caliper)
# ──────────────────────────────────────────────────────────────────────────────

def feret_diameters(contour: np.ndarray) -> Tuple[float, float]:
    """
    Compute Feret max and min via rotating caliper over the convex hull.
    Returns (feret_max_px, feret_min_px).
    """
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(float)
    n = len(hull)
    if n < 2:
        return 0.0, 0.0

    max_d, min_d = 0.0, float("inf")
    angles = np.linspace(0, math.pi, 180)

    for theta in angles:
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        projections = hull[:, 0] * cos_t + hull[:, 1] * sin_t
        d = projections.max() - projections.min()
        max_d = max(max_d, d)
        min_d = min(min_d, d)

    return max_d, min_d


# ──────────────────────────────────────────────────────────────────────────────
# 5. Angularity index
# ──────────────────────────────────────────────────────────────────────────────

def angularity_index(contour: np.ndarray, n_points: int = 32) -> float:
    """
    Normalised angularity index (0 = perfect circle, 1 = very angular).
    Computed as the mean absolute angular deviation along the resampled contour.
    Reference: Wadell (1932) reworked for digital image analysis.
    """
    pts = contour.reshape(-1, 2).astype(float)
    if len(pts) < 6:
        return 0.0

    # Resample to n_points equally spaced along the contour
    arc = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    total = arc[-1]
    if total < 1e-6:
        return 0.0
    sample_arc = np.linspace(0, total, n_points, endpoint=False)
    sx = np.interp(sample_arc, arc, pts[:, 0])
    sy = np.interp(sample_arc, arc, pts[:, 1])
    sampled = np.column_stack([sx, sy])

    # Turning angles
    v1 = np.roll(sampled, -1, axis=0) - sampled
    v2 = np.roll(sampled, -2, axis=0) - np.roll(sampled, -1, axis=0)
    dot   = np.einsum("ij,ij->i", v1, v2)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    angles = np.abs(np.arctan2(cross, dot))

    # Normalise by expected circle deviation (≈ 2π / n_points * 2)
    expected = 2 * math.pi / n_points
    ai = float(np.mean(angles) / (expected + 1e-9)) - 1.0
    return max(0.0, min(ai, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# 6. Texture – GLCM (pure NumPy)
# ──────────────────────────────────────────────────────────────────────────────

def _glcm_matrix(patch: np.ndarray, levels: int = 16, dx: int = 1, dy: int = 0) -> np.ndarray:
    """Compute normalised GLCM for a single offset."""
    q = (patch.astype(np.float32) * (levels - 1) / 255).astype(np.int32)
    q = np.clip(q, 0, levels - 1)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    h, w = q.shape
    r_start = max(0, -dy);  r_end = h - max(0, dy)
    c_start = max(0, -dx);  c_end = w - max(0, dx)
    src = q[r_start:r_end, c_start:c_end]
    dst = q[r_start + dy:r_end + dy, c_start + dx:c_end + dx]
    for i, j in zip(src.ravel(), dst.ravel()):
        glcm[i, j] += 1
    glcm += glcm.T                  # symmetric
    total = glcm.sum()
    return glcm / total if total > 0 else glcm


def glcm_features(patch: np.ndarray, levels: int = 16) -> Dict[str, float]:
    """Return contrast, correlation, energy, homogeneity averaged over 4 offsets."""
    offsets = [(1, 0), (0, 1), (1, 1), (1, -1)]
    results = {"contrast": [], "correlation": [], "energy": [], "homogeneity": []}

    for dx, dy in offsets:
        g = _glcm_matrix(patch, levels, dx, dy)
        i_idx, j_idx = np.mgrid[0:levels, 0:levels]

        contrast    = float(np.sum(g * (i_idx - j_idx) ** 2))
        energy      = float(np.sum(g ** 2))
        homogeneity = float(np.sum(g / (1 + np.abs(i_idx - j_idx))))

        mu_i  = float(np.sum(i_idx * g))
        mu_j  = float(np.sum(j_idx * g))
        sig_i = math.sqrt(max(float(np.sum(g * (i_idx - mu_i) ** 2)), 1e-9))
        sig_j = math.sqrt(max(float(np.sum(g * (j_idx - mu_j) ** 2)), 1e-9))
        correlation = float(np.sum(g * (i_idx - mu_i) * (j_idx - mu_j)) / (sig_i * sig_j))

        results["contrast"].append(contrast)
        results["correlation"].append(correlation)
        results["energy"].append(energy)
        results["homogeneity"].append(homogeneity)

    return {k: float(np.mean(v)) for k, v in results.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 7. Texture – Wavelet energy
# ──────────────────────────────────────────────────────────────────────────────

def wavelet_energy(patch: np.ndarray, wavelet: str = "db4", level: int = 2) -> Dict[str, float]:
    """Daubechies wavelet decomposition; returns approx + detail energies."""
    if not _HAS_PYWT:
        return {"wavelet_energy_approx": 0.0, "wavelet_energy_detail": 0.0}
    coeffs = pywt.wavedec2(patch.astype(float), wavelet, level=level)
    approx_energy = float(np.sum(coeffs[0] ** 2))
    detail_energy = float(sum(np.sum(c ** 2) for cset in coeffs[1:] for c in cset))
    return {
        "wavelet_energy_approx": approx_energy,
        "wavelet_energy_detail": detail_energy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Per-particle measurement
# ──────────────────────────────────────────────────────────────────────────────

def measure_particle(
    particle_id: int,
    contour: np.ndarray,
    gray: np.ndarray,
    mm_per_px: float,
    min_area_px: int = 200,
) -> Optional[ParticleMetrics]:
    """Extract all metrics for a single particle contour."""
    area_px  = cv2.contourArea(contour)
    peri_px  = cv2.arcLength(contour, closed=True)

    if area_px < min_area_px or peri_px < 1:
        return None

    # Sub-pixel moments
    M = cv2.moments(contour)
    if M["m00"] < 1e-6:
        return None

    # Bounding rect for texture patch
    x, y, bw, bh = cv2.boundingRect(contour)
    patch = gray[y:y + bh, x:x + bw]

    # Feret
    feret_max_px, feret_min_px = feret_diameters(contour)

    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    # Shape metrics
    circularity   = (4 * math.pi * area_px / (peri_px ** 2)) if peri_px > 0 else 0.0
    elongation    = (feret_max_px / feret_min_px) if feret_min_px > 1 else 1.0
    convexity     = (hull_area / area_px) if area_px > 0 else 1.0
    solidity      = (area_px / hull_area) if hull_area > 0 else 1.0
    ai            = angularity_index(contour)

    # Scale conversions
    def px_to_mm(px: float) -> float:
        return px * mm_per_px

    def px2_to_mm2(px2: float) -> float:
        return px2 * mm_per_px ** 2

    equiv_diam_mm = px_to_mm(2 * math.sqrt(area_px / math.pi))

    # Texture
    glcm  = glcm_features(patch) if patch.size > 16 else {}
    wavel = wavelet_energy(patch) if patch.size > 16 else {}

    # Confidence: penalise near-border, very small, or abnormal circularity
    h_img, w_img = gray.shape[:2]
    border_margin = 5
    on_border = (x <= border_margin or y <= border_margin or
                 x + bw >= w_img - border_margin or
                 y + bh >= h_img - border_margin)
    conf = 1.0
    if on_border:
        conf *= 0.5
    if circularity > 1.05:   # impossible → segmentation artefact
        conf *= 0.3
    if area_px < min_area_px * 2:
        conf *= 0.7

    return ParticleMetrics(
        particle_id=particle_id,
        area_mm2=          px2_to_mm2(area_px),
        perimeter_mm=      px_to_mm(peri_px),
        equiv_diameter_mm= equiv_diam_mm,
        feret_max_mm=      px_to_mm(feret_max_px),
        feret_min_mm=      px_to_mm(feret_min_px),
        elongation=        elongation,
        circularity=       min(circularity, 1.0),
        convexity=         convexity,
        angularity_index=  ai,
        solidity=          solidity,
        glcm_contrast=     glcm.get("contrast",    0.0),
        glcm_correlation=  glcm.get("correlation", 0.0),
        glcm_energy=       glcm.get("energy",      0.0),
        glcm_homogeneity=  glcm.get("homogeneity", 0.0),
        wavelet_energy_approx= wavel.get("wavelet_energy_approx", 0.0),
        wavelet_energy_detail= wavel.get("wavelet_energy_detail", 0.0),
        confidence=        conf,
        outlier_flag=      conf < 0.6,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9. Outlier detection (IQR fence on key metrics)
# ──────────────────────────────────────────────────────────────────────────────

def flag_outliers(particles: List[ParticleMetrics], k: float = 2.5) -> None:
    """Flag particles whose area or elongation is outside k × IQR of the group."""
    for metric in ("area_mm2", "elongation"):
        vals = np.array([getattr(p, metric) for p in particles], dtype=float)
        if len(vals) < 4:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        for p, v in zip(particles, vals):
            if v < lo or v > hi:
                p.outlier_flag = True
                p.confidence   = min(p.confidence, 0.4)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Main pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────

def analyse_image(
    image: np.ndarray,
    calib: Optional[CalibrationResult] = None,
    mm_per_px: float = 0.1,
    use_watershed: bool = True,
    min_area_px: int = 200,
) -> AnalysisResult:
    """
    Full AIMS pipeline.

    Parameters
    ----------
    image        : BGR or greyscale numpy array (uint8)
    calib        : CalibrationResult from calibrate_from_checkerboard()
                   or calibrate_scalar(); if None, uses mm_per_px directly.
    mm_per_px    : fallback scale when calib is None
    use_watershed: True = watershed segmentation (precise), False = simple Otsu
    min_area_px  : smallest particle accepted (pixels)

    Returns
    -------
    AnalysisResult with per-particle metrics and summary statistics
    """
    if calib is None:
        calib = calibrate_scalar(mm_per_px)

    # Undistort (no-op if no camera matrix)
    img_rect = undistort_image(image, calib)

    # Preprocess
    gray = preprocess(img_rect)

    # Segment
    if use_watershed:
        _, label_map = segment_watershed(gray, min_area_px)
    else:
        _, label_map = segment_simple(gray, min_area_px)

    # Measure each particle
    particles: List[ParticleMetrics] = []
    rejected = 0
    pid = 0

    unique_labels = np.unique(label_map)
    unique_labels = unique_labels[unique_labels > 1]   # skip bg (0) and border (1)

    for lbl in unique_labels:
        mask = (label_map == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        metrics = measure_particle(pid, cnt, gray, calib.mm_per_px, min_area_px)
        if metrics is None:
            rejected += 1
            continue
        particles.append(metrics)
        pid += 1

    flag_outliers(particles)

    return AnalysisResult(
        particles=particles,
        calibration=calib,
        total_particles=len(particles),
        rejected_particles=rejected,
        image_shape=gray.shape,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 11. Quick synthetic self-test  (run: python -m aims.processing)
# ──────────────────────────────────────────────────────────────────────────────

def _synthetic_test() -> None:
    print("=" * 60)
    print("AIMS processing.py  –  synthetic self-test")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Create a 512×512 white image with 8 dark ellipses
    img = np.full((512, 512, 3), 240, dtype=np.uint8)
    particles_gt = [
        {"cx": 80,  "cy": 80,  "a": 35, "b": 20, "angle": 30},
        {"cx": 200, "cy": 80,  "a": 28, "b": 28, "angle": 0},
        {"cx": 340, "cy": 90,  "a": 45, "b": 18, "angle": 60},
        {"cx": 80,  "cy": 250, "a": 30, "b": 22, "angle": 15},
        {"cx": 220, "cy": 240, "a": 25, "b": 25, "angle": 0},
        {"cx": 380, "cy": 260, "a": 40, "b": 15, "angle": 80},
        {"cx": 130, "cy": 400, "a": 32, "b": 24, "angle": 45},
        {"cx": 360, "cy": 400, "a": 28, "b": 20, "angle": 10},
    ]

    for p in particles_gt:
        cv2.ellipse(
            img,
            (p["cx"], p["cy"]),
            (p["a"], p["b"]),
            p["angle"], 0, 360,
            (60, 60, 60), -1
        )
        # Add subtle texture inside
        noise_patch = rng.integers(40, 90, (p["b"] * 2, p["a"] * 2, 3), dtype=np.uint8)
        x1, y1 = max(0, p["cx"] - p["a"]), max(0, p["cy"] - p["b"])
        x2, y2 = min(512, p["cx"] + p["a"]), min(512, p["cy"] + p["b"])
        img[y1:y2, x1:x2] = noise_patch[:y2-y1, :x2-x1]

    # Add Gaussian noise to entire image
    noise = rng.integers(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Run analysis  (1 px = 0.05 mm)
    calib  = calibrate_scalar(mm_per_px=0.05)
    result = analyse_image(img, calib=calib, use_watershed=True, min_area_px=100)

    print(f"\nDetected particles  : {result.total_particles}")
    print(f"Rejected (too small): {result.rejected_particles}")
    print(f"Expected            : ~{len(particles_gt)}")

    print("\nPer-particle summary:")
    print(f"  {'ID':>4}  {'Area mm²':>9}  {'Feret max':>9}  {'Elong':>6}  "
          f"{'Circ':>5}  {'Ang':>5}  {'Conf':>5}  {'Outlier':>7}")
    for p in result.particles:
        print(f"  {p.particle_id:>4}  {p.area_mm2:>9.4f}  "
              f"{p.feret_max_mm:>9.4f}  {p.elongation:>6.2f}  "
              f"{p.circularity:>5.3f}  {p.angularity_index:>5.3f}  "
              f"{p.confidence:>5.2f}  {'YES' if p.outlier_flag else 'no':>7}")

    print("\nAggregate statistics (valid particles only):")
    stats = result.summary_stats()
    for metric, vals in stats.items():
        print(f"  {metric:<26} mean={vals['mean']:.4f}  std={vals['std']:.4f}  "
              f"min={vals['min']:.4f}  max={vals['max']:.4f}")

    print("\n✓ Self-test complete – processing.py is functional.")


if __name__ == "__main__":
    _synthetic_test()
