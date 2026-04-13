import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from distortion_compensation_v2 import compensate_distortion


def warp_spherical(img, focal_length):
    """Applies spherical projection."""
    height, width = img.shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)

    x_c = x - width / 2
    y_c = y - height / 2

    theta = x_c / focal_length
    phi = y_c / focal_length

    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(phi)
    Z = np.cos(theta) * np.cos(phi)

    u = focal_length * X / Z + width / 2
    v = focal_length * Y / Z + height / 2

    return cv2.remap(img, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def prepare_dataset(path_base, dataset_name, bool_spherical_wraped=True):
    path_base = Path(path_base)
    # Check if images are inside a subfolder named after the dataset
    input_dir = path_base / dataset_name

    # If the subfolder doesn't exist, fallback to base path
    if not input_dir.exists():
        input_dir = path_base
        print(f"[Info] Subfolder {dataset_name} not found, searching in {path_base}")

    timestamp = datetime.now().strftime("%d_%m__at__%Hh_%Mm_%Ss")
    temp_dir = path_base / f"temp_{dataset_name}_{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Start] Processing images from: {input_dir.absolute()}")

    for i in range(1, 9):
        img_name = f"Cam{i}_hall.jpg"
        input_path = input_dir / img_name

        # Check if file exists before processing
        if not input_path.exists():
            print(f"[Error] File not found: {input_path.absolute()}")
            continue

        # 1. Distortion compensation
        try:
            compensate_distortion(str(input_path), str(temp_dir))
        except Exception as e:
            print(f"[Error] Failed compensation for {img_name}: {e}")
            continue

        # 2. Spherical Warp (Overwrite compensated image)
        comp_img_path = temp_dir / f"Cam{i}_hall_compensated.jpg"

        if bool_spherical_wraped and comp_img_path.exists():
            img = cv2.imread(str(comp_img_path))
            if img is not None:
                f = img.shape[1] * 0.8  # Focal length adjustment
                warped = warp_spherical(img, f)
                cv2.imwrite(str(comp_img_path), warped)
                print(f"[Success] {img_name} processed.")
            else:
                print(f"[Error] Could not read compensated image: {comp_img_path}")

    return temp_dir


if __name__ == "__main__":
    # Use raw string (r"") for Windows paths
    BASE_PATH = r"C:\Users\lmori\anaconda_projects\Smart helmet\data"
    prepare_dataset(BASE_PATH, "ds1", False)