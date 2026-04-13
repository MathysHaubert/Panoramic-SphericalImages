import cv2
import numpy as np
from pathlib import Path


def _save_result(image, original_path, suffix, out_dir=None):
    """Helper to save images with a specific method suffix in a target directory."""
    original_path = Path(original_path)
    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        output_path = out_path / f"{original_path.stem}_{suffix}.jpg"
    else:
        output_path = original_path.with_name(f"{original_path.stem}_{suffix}.jpg")

    cv2.imwrite(str(output_path), image)
    print(f"Saved: {output_path}")


def compensate_grakovski(image_path, K=0.3, lambda_val=100.0, phi=-1.06, preserve_fov=True, out_dir=None):
    """
    Model based on Grakovski paper with exponential k(r).
    Formula: k(r) = K + exp(lambda * (r + phi))
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    x_dst, y_dst = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_n, y_n = (x_dst - cx) / max(cx, 1.0), (y_dst - cy) / max(cy, 1.0)

    r2 = x_n ** 2 + y_n ** 2
    r_norm = np.sqrt(r2) / np.sqrt(2.0)

    k_r = K + np.exp(lambda_val * (r_norm + phi))

    p = 2.0 * k_r * ((r2 / 2.0) - (r2 ** 2 / 24.0) + (r2 ** 3 / 720.0) - (r2 ** 4 / 40320.0))
    g = 1.0 / np.clip(1.0 + p, 1e-6, None)

    x_src_n, y_src_n = x_n * g, y_n * g

    if preserve_fov:
        scale = 1.0 / float(max(np.max(np.abs(x_src_n)), np.max(np.abs(y_src_n))))
        x_src_n, y_src_n = x_src_n * scale, y_src_n * scale

    map_x, map_y = x_src_n * max(cx, 1.0) + cx, y_src_n * max(cy, 1.0) + cy
    corrected = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

    _save_result(corrected, image_path, "compensated_grakovski", out_dir)


def compensate_sigmoid(image_path, K_center=0.3, K_edge=0.26, alpha=20.0, r0=0.85, preserve_fov=True, out_dir=None):
    """
    Sigmoid model to smoothly transition correction strength.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    x_dst, y_dst = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_n, y_n = (x_dst - cx) / max(cx, 1.0), (y_dst - cy) / max(cy, 1.0)

    r2 = x_n ** 2 + y_n ** 2
    r_norm = np.sqrt(r2) / np.sqrt(2.0)

    k_r = K_center + (K_edge - K_center) / (1.0 + np.exp(-alpha * (r_norm - r0)))

    p = 2.0 * k_r * ((r2 / 2.0) - (r2 ** 2 / 24.0) + (r2 ** 3 / 720.0) - (r2 ** 4 / 40320.0))
    g = 1.0 / np.clip(1.0 + p, 1e-6, None)

    x_src_n, y_src_n = x_n * g, y_n * g

    if preserve_fov:
        scale = 1.0 / float(max(np.max(np.abs(x_src_n)), np.max(np.abs(y_src_n))))
        x_src_n, y_src_n = x_src_n * scale, y_src_n * scale

    map_x, map_y = x_src_n * max(cx, 1.0) + cx, y_src_n * max(cy, 1.0) + cy
    corrected = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

    _save_result(corrected, image_path, "compensated", out_dir)


def compensate_opencv(image_path, k1=-0.06, k2=0.0, out_dir=None):
    """
    Standard OpenCV Fisheye implementation.
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    f = w * 0.4
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    D = np.array([k1, k2, 0, 0])

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    corrected = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    _save_result(corrected, image_path, "compensated_opencv", out_dir)


def run_all_methods(image_path, out_dir="output"):
    """Run all methods and save results in the specified folder."""
    print(f"Processing: {image_path} into {out_dir}/")
    compensate_grakovski(image_path, out_dir=out_dir)
    compensate_sigmoid(image_path, out_dir=out_dir)
    compensate_opencv(image_path, out_dir=out_dir)


def compensate_distortion(image_path, out_dir="output"):
    """Run the best methods with their parameters and save results in the specified folder."""
    K_center = 0.30
    K_edge = 0.25
    alpha = 40
    r0 = 0.85
    preserve_fov = True
    compensate_sigmoid(image_path, K_center, K_edge, alpha, r0, preserve_fov, out_dir)