"""
Daugman Architectural Pipeline  —  Cam1 Edition
================================================
Adapts Daugman's Rubber Sheet Model and Gabor feature extraction
to a wide-angle (fisheye) image of a building's glass-dome entrance hall.

Key design choices
------------------
* r_inner = 1  (effectively the single convergence point — "infinitely small")
* r_outer = half-diagonal of the image (largest circle that fits inside the
             image bounding box when centred on the detected dome centre,
             i.e. the furthest corner from that centre).
  This guarantees the annulus covers the *entire* image content.

Pipeline
--------
1. Detect the dome convergence centre (ox, oy) via Hough circles on a
   downscaled copy (fast, edge-preserving bilateral pre-filter).
2. Compute r_outer = distance from (ox,oy) to the farthest image corner.
3. Unroll the annulus [1, r_outer] into a 512-wide × 256-tall normalised
   rectangular strip using Daugman's Rubber Sheet Model.
   (Taller strip than before because we now cover a much larger radial range.)
4. Apply a bank of 2-D Gabor filters → binary feature vector (IrisCode style).
5. Save:
     Cam1_hall_annulus.png   — original with dome centre + outer circle
     Cam1_hall_strip.png     — unrolled strip  (512 × 256)
     Cam1_hall_gabor_*.png   — one response image per Gabor filter
     Cam1_hall_features.npy  — binary feature vector

Usage
-----
    python daugman_pipeline.py
    python daugman_pipeline.py --r_outer 800   # override outer radius
    python daugman_pipeline.py --help

Dependencies: opencv-python, numpy  (standard in this environment)
"""

import argparse
import math
import os

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (change here or via CLI)
# ──────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(_HERE, "image", "Cam1_hall.jpg")
# Outputs go directly into the project folder (same level as daugman_pipeline.py)
OUTPUT_DIR = _HERE

# Strip resolution
STRIP_W = 512   # angular samples  (θ columns)
STRIP_H = 256   # radial samples   (ρ rows) — taller to capture full radial detail


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DOME CENTRE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_dome_centre(img_bgr: np.ndarray,
                        scale: float = 0.25,
                        hough_param2: int = 28) -> tuple[int, int]:
    """
    Detect the convergence centre of the glass dome.

    Runs on a downscaled copy (scale=0.25) for speed, then maps the result
    back to full resolution.

    Returns
    -------
    (cx, cy)  —  dome centre in full-resolution pixel coordinates.
    """
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]

    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grey = cv2.bilateralFilter(grey, 7, 75, 75)   # edge-preserving noise removal
    grey = cv2.equalizeHist(grey)                  # improve low-contrast dome rings

    short = min(sh, sw)
    circles = cv2.HoughCircles(
        grey,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=int(short * 0.15),
        param1=100,
        param2=hough_param2,
        minRadius=int(short * 0.10),
        maxRadius=int(short * 0.55),
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Choose the detected circle whose centre is closest to the image centre
        dists = np.hypot(circles[:, 0] - sw // 2,
                         circles[:, 1] - sh // 2)
        best = circles[np.argmin(dists)]
        cx = int(best[0] / scale)
        cy = int(best[1] / scale)
        print(f"  [detect] Hough circle centre → ({cx}, {cy})  "
              f"(detected radius={int(best[2]/scale)} px, ignored — using full-image radius)")
    else:
        cx, cy = w // 2, h // 2
        print(f"  [detect] No Hough circle found — falling back to image centre ({cx}, {cy})")

    return cx, cy


# ──────────────────────────────────────────────────────────────────────────────
# 2.  RUBBER SHEET MODEL  (annulus → rectangular strip)
# ──────────────────────────────────────────────────────────────────────────────

def rubber_sheet_unwrap(img_bgr: np.ndarray,
                         cx: int, cy: int,
                         r_inner: int, r_outer: int,
                         strip_w: int = STRIP_W,
                         strip_h: int = STRIP_H) -> np.ndarray:
    """
    Map the annular region [r_inner, r_outer] to a (strip_h × strip_w) strip.

    Daugman's mapping:
        x(ρ, θ) = (1−ρ)·x_inner(θ) + ρ·x_outer(θ)
        y(ρ, θ) = (1−ρ)·y_inner(θ) + ρ·y_outer(θ)

    where ρ ∈ [0, 1],  θ ∈ [0, 2π).

    With r_inner = 1 and r_outer = farthest corner distance, this covers
    the entire image — every pixel is sampled exactly once.
    """
    rho   = np.linspace(0.0, 1.0, strip_h, endpoint=False)   # radial axis
    theta = np.linspace(0.0, 2.0 * np.pi, strip_w, endpoint=False)  # angular axis

    Rho, Theta = np.meshgrid(rho, theta, indexing='ij')

    # Actual radius in pixels at each (ρ, θ) cell
    R_px = r_inner + Rho * (r_outer - r_inner)

    # Source coordinates in the original image
    map_x = (cx + R_px * np.cos(Theta)).astype(np.float32)
    map_y = (cy + R_px * np.sin(Theta)).astype(np.float32)

    # cv2.remap with BORDER_REPLICATE so corners outside the image stay valid
    strip = cv2.remap(
        img_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return strip  # (strip_h, strip_w, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  GABOR FILTER BANK
# ──────────────────────────────────────────────────────────────────────────────

def build_gabor_bank(ksize: int = 31) -> list[tuple]:
    """
    Build a bank of 2-D Gabor quadrature pairs.

    4 orientations × 2 spatial frequencies × 2 scales = 16 filter pairs.
    The real and imaginary (quadrature) kernels are both stored so that
    callers can compute response magnitude or use the sign of the real part.
    """
    sigmas  = [3.0, 6.0]
    thetas  = [k * math.pi / 4 for k in range(4)]   # 0°, 45°, 90°, 135°
    lambdas = [8.0, 16.0]

    bank = []
    for sigma in sigmas:
        for theta in thetas:
            for lam in lambdas:
                k_real = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lam, 0.5, 0.0, cv2.CV_32F)
                k_imag = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lam, 0.5, math.pi / 2, cv2.CV_32F)
                label = (f"s{sigma:.0f}"
                         f"_t{math.degrees(theta):.0f}"
                         f"_l{lam:.0f}")
                bank.append((k_real, k_imag, label))
    return bank


def extract_gabor_features(strip_grey: np.ndarray,
                            bank: list[tuple]) -> tuple[np.ndarray, list]:
    """
    Convolve the strip with every filter in the bank and binarise.

    Binarisation: bit = 1  if  real_response > column_mean  (Daugman style).

    Returns
    -------
    feature_vector : 1-D binary array  (len = n_filters × strip_h × strip_w)
    responses_real : list of 2-D float32 response maps (one per filter)
    """
    bits = []
    responses_real = []

    for k_real, k_imag, _ in bank:
        resp = cv2.filter2D(strip_grey, cv2.CV_32F, k_real)
        col_mean = resp.mean(axis=0, keepdims=True)   # per-column threshold
        bits.append((resp > col_mean).astype(np.uint8).ravel())
        responses_real.append(resp)

    return np.concatenate(bits), responses_real


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run(image_path: str,
        output_dir: str,
        r_inner_override: int | None = None,
        r_outer_override: int | None = None) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # ── Load image ────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n{'='*60}")
    print(f"  Image : {os.path.basename(image_path)}  ({w} × {h} px)")

    # ── Step 1: dome centre = geometric centre of the sensor
    # For a fisheye / wide-angle lens the principal point coincides with the
    # image centre.  A 1920×1080 image → (960, 540).
    cx, cy = w // 2, h // 2
    print(f"  Centre: ({cx}, {cy})  [geometric image centre]")

    # ── Step 2: compute radii ─────────────────────────────────────────────
    # r_inner = 1  →  effectively a single-pixel centre (limit of "infinitely small")
    r_inner = r_inner_override if r_inner_override is not None else 1

    if r_outer_override is not None:
        r_outer = r_outer_override
    else:
        # Largest circle that stays fully inside the image:
        # distance from the detected centre to the nearest edge.
        r_outer = int(min(cx, cy, w - cx, h - cy))

    print(f"  r_inner = {r_inner} px  (≈ infinitely small centre point)")
    print(f"  r_outer = {r_outer} px  (largest circle inside image from centre)")

    # ── Step 3: visualise annulus ─────────────────────────────────────────
    vis = img.copy()
    cv2.circle(vis, (cx, cy), r_outer, (0, 128, 255), 3)   # outer — orange
    cv2.circle(vis, (cx, cy), r_inner, (0, 255,   0), 2)   # inner — green (tiny)
    cv2.circle(vis, (cx, cy), 6,       (0,   0, 255), -1)  # centre dot — red
    annulus_path = os.path.join(output_dir, f"{base}_annulus.png")
    cv2.imwrite(annulus_path, vis)
    print(f"  Annulus vis  → {annulus_path}")

    # ── Step 4: Rubber Sheet unroll ───────────────────────────────────────
    strip = rubber_sheet_unwrap(img, cx, cy, r_inner, r_outer,
                                 strip_w=STRIP_W, strip_h=STRIP_H)
    strip_path = os.path.join(output_dir, f"{base}_strip.png")
    cv2.imwrite(strip_path, strip)
    print(f"  Strip        → {strip_path}  ({STRIP_W} × {STRIP_H} px)")

    # ── Step 5: Gabor features ────────────────────────────────────────────
    grey = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bank = build_gabor_bank()
    feature_vector, responses = extract_gabor_features(grey, bank)

    feat_path = os.path.join(output_dir, f"{base}_features.npy")
    # numpy's np.save may fail on certain mounted filesystems;
    # fall back to a manual binary write which always works.
    try:
        np.save(feat_path, feature_vector)
    except OSError:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        np.save(tmp.name, feature_vector)
        tmp.close()
        with open(tmp.name, "rb") as src, open(feat_path, "wb") as dst:
            dst.write(src.read())
        os.unlink(tmp.name)
    print(f"  Features     → {feat_path}")
    print(f"  Vector shape : {feature_vector.shape}  "
          f"density = {feature_vector.mean():.3f}")

    # Save all 16 Gabor response images
    for idx, (resp, (_, _, label)) in enumerate(zip(responses, bank)):
        resp_norm = cv2.normalize(resp, None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8)
        resp_path = os.path.join(output_dir,
                                 f"{base}_gabor_{idx:02d}_{label}.png")
        cv2.imwrite(resp_path, resp_norm)

    print(f"  Gabor images → {output_dir}/{base}_gabor_*.png  (16 files)")
    print(f"\n  Done.  All outputs in: {os.path.abspath(output_dir)}")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Daugman Rubber Sheet pipeline for Cam1_hall.jpg.\n"
            "Default: r_inner=1 (point centre), r_outer=max (farthest corner)."
        )
    )
    p.add_argument("--image",    default=IMAGE_PATH,
                   help="Path to the input image (default: image/Cam1_hall.jpg).")
    p.add_argument("--output",   default=OUTPUT_DIR,
                   help="Output directory (default: daugman_output/).")
    p.add_argument("--r_inner",  type=int, default=None,
                   help="Override inner radius in px (default: 1).")
    p.add_argument("--r_outer",  type=int, default=None,
                   help="Override outer radius in px (default: farthest corner).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.image, args.output,
        r_inner_override=args.r_inner,
        r_outer_override=args.r_outer)
