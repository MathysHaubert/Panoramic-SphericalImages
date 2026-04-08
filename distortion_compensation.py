import cv2
import numpy as np
from pathlib import Path


def compensate_distortion(image_path, k=0.3, preserve_fov=True, output_dir=None):
    """
    Compensate radial distortion using a single-parameter exponential model.
    Accepts an output_dir to save the result directly in a specific folder.
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    h, w = image.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    x_dst, y_dst = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_n = (x_dst - cx) / max(cx, 1.0)
    y_n = (y_dst - cy) / max(cy, 1.0)

    r2 = x_n * x_n + y_n * y_n
    p = 2.0 * k * ((r2 / 2.0) - ((r2 * r2) / 24.0) + ((r2 * r2 * r2) / 720.0) - ((r2 * r2 * r2 * r2) / 40320.0))
    g = 1.0 / np.clip(1.0 + p, 1e-6, None)

    x_src_n = x_n * g
    y_src_n = y_n * g

    if preserve_fov:
        max_abs = float(max(np.max(np.abs(x_src_n)), np.max(np.abs(y_src_n))))
        if max_abs > 1e-6:
            scale = 1.0 / max_abs
            x_src_n *= scale
            y_src_n *= scale

    map_x = x_src_n * max(cx, 1.0) + cx
    map_y = y_src_n * max(cy, 1.0) + cy

    corrected = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32),
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Gestion du dossier de sortie
    if output_dir is not None:
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = out_dir_path / f"{image_path.stem}_compensated.jpg"
    else:
        output_path = image_path.with_name(f"{image_path.stem}_compensated.jpg")

    cv2.imwrite(str(output_path), corrected)
    return str(output_path)