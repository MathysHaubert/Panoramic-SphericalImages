
import os
import threading
import traceback
import tkinter
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

from gui.windows.ask_path import TopWindow


MATCH_SIZE = 400
COMPOSE_SIZE = 1200
OUT_W = 5000
OUT_H = 2500
CROP_PCT = 0.76

PAIRS = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1)
]


def resize_keep(img, ms):
    h, w = img.shape[:2]
    s = ms / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)))


def safe_imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def safe_imwrite(path, img, quality=94):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == ".png":
        ok, buf = cv2.imencode(".png", img)
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def bundle_adjustment(imgs, raw, log=print):
    n = len(imgs)
    log(f"\n[1/3] Bundle adjustment ({MATCH_SIZE}px)…")
    sift = cv2.SIFT_create(6000)
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    kps_all, des_all = zip(*[sift.detectAndCompute(g, None) for g in grays])
    all_matches = {}

    for (i, j) in PAIRS:
        if i >= n or j >= n:
            continue
        if des_all[i] is None or des_all[j] is None:
            continue

        raw_m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_all[i], des_all[j], k=2)
        good = [m for m, n2 in raw_m if m.distance < 0.70 * n2.distance]
        if len(good) < 8:
            continue

        p1 = np.float32([kps_all[i][m.queryIdx].pt for m in good])
        p2 = np.float32([kps_all[j][m.trainIdx].pt for m in good])

        _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
        if mask is None:
            continue
        mask = mask.ravel().astype(bool)
        if mask.sum() < 8:
            continue

        all_matches[(i, j)] = (p1[mask], p2[mask])
        log(f"  cam{i+1}↔cam{j+1}: {mask.sum()} inliers")

    imgs300 = [resize_keep(img, 300) for img in raw]
    finder = cv2.SIFT_create(4000)
    feats = [cv2.detail.computeImageFeatures2(finder, img) for img in imgs300]
    mat = cv2.detail.BestOf2NearestMatcher_create(False, 0.3)
    mts = mat.apply2(feats)
    mat.collectGarbage()
    _, cams = cv2.detail.HomographyBasedEstimator().apply(feats, mts, None)

    focal_init = sorted([abs(c.focal) for c in cams])[n // 2] * (MATCH_SIZE / 300)
    quats_init = [Rot.from_matrix(c.R.astype(np.float64)).as_quat() for c in cams]

    def unproject(pts, f, cx, cy):
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        r = np.sqrt(dx**2 + dy**2)
        theta = r / f
        phi = np.arctan2(dy, dx)
        st = np.sin(theta)
        rays = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
        nr = np.linalg.norm(rays, axis=1, keepdims=True)
        return rays / np.where(nr > 0, nr, 1)

    def pack(f, qs):
        return np.concatenate([[f]] + [q for q in qs[1:]])

    def unpack(x):
        f = x[0]
        qs = [np.array([0, 0, 0, 1], np.float64)]
        for k in range(n - 1):
            q = x[1 + k * 4:1 + k * 4 + 4]
            qs.append(q / max(np.linalg.norm(q), 1e-12))
        return f, qs

    def cost(x):
        f, qs = unpack(x)
        if f < 30 or f > 3000:
            return 1e8
        total, count = 0.0, 0
        for (i, j), (p1, p2) in all_matches.items():
            h1, w1 = imgs[i].shape[:2]
            h2, w2 = imgs[j].shape[:2]
            r1 = unproject(p1, f, w1 / 2, h1 / 2)
            r2 = unproject(p2, f, w2 / 2, h2 / 2)
            R1 = Rot.from_quat(qs[i]).as_matrix()
            R2 = Rot.from_quat(qs[j]).as_matrix()
            w1r = (R1 @ r1.T).T
            w2r = (R2 @ r2.T).T
            dots = np.clip((w1r * w2r).sum(axis=1), -1, 1)
            total += np.arccos(np.abs(dots)).sum()
            count += len(dots)
        return total / max(count, 1)

    x0 = pack(focal_init, quats_init)
    log(f"  Coût initial : {cost(x0) * 1000:.1f} mrad")
    res = minimize(cost, x0, method='Powell',
                   options={'maxiter': 8000, 'ftol': 1e-10, 'xtol': 1e-10, 'disp': False})
    f_opt, q_opt = unpack(res.x)
    log(f"  Coût final   : {res.fun * 1000:.2f} mrad  (f={f_opt:.1f}px)")
    return f_opt, q_opt


def project_sphere(raw, f_ba, quats, ba_size, log=print):
    log(f"\n[2/3] Projection sphérique ({COMPOSE_SIZE}px → {OUT_W}×{OUT_H})…")
    imgs_c = [resize_keep(img, COMPOSE_SIZE) for img in raw]
    f_c = f_ba * (COMPOSE_SIZE / ba_size)
    lon = np.linspace(-np.pi, np.pi, OUT_W, dtype=np.float32)
    lat = np.linspace(np.pi / 2, -np.pi / 2, OUT_H, dtype=np.float32)
    LON, LAT = np.meshgrid(lon, lat)
    world = np.stack([(np.cos(LAT) * np.cos(LON)).ravel(),
                      (np.cos(LAT) * np.sin(LON)).ravel(),
                      np.sin(LAT).ravel()], axis=1)
    samplers = []
    masks = []
    theta_maps = []

    for i, (img, q) in enumerate(zip(imgs_c, quats)):
        h, w = img.shape[:2]
        cx, cy = w / 2., h / 2.
        hf = np.arctan2(np.sqrt((w / 2) ** 2 + (h / 2) ** 2), f_c)
        R = Rot.from_quat(q).as_matrix()
        pts = world @ R
        Xc, Yc, Zc = pts[:, 0], pts[:, 1], pts[:, 2]
        rxy = np.sqrt(Xc**2 + Yc**2)
        theta = np.arctan2(rxy, Zc)
        valid = theta < hf
        phi = np.arctan2(Yc, Xc)
        rpx = f_c * theta
        u = np.where(valid, cx + rpx * np.cos(phi), -1).astype(np.float32).reshape(OUT_H, OUT_W)
        v = np.where(valid, cy + rpx * np.sin(phi), -1).astype(np.float32).reshape(OUT_H, OUT_W)
        ib = (
            valid
            & (u.ravel() >= 0) & (u.ravel() < w - 1)
            & (v.ravel() >= 0) & (v.ravel() < h - 1)
        ).reshape(OUT_H, OUT_W)
        samp = cv2.remap(img, u, v, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        samplers.append(samp)
        masks.append(ib)
        theta_maps.append(theta.reshape(OUT_H, OUT_W).astype(np.float32))
        log(f"  cam{i+1}: {ib.sum():,} px")

    best_theta = np.full((OUT_H, OUT_W), np.inf, np.float32)
    winner = np.full((OUT_H, OUT_W), -1, np.int8)
    for i in range(len(imgs_c)):
        better = masks[i] & (theta_maps[i] < best_theta)
        best_theta[better] = theta_maps[i][better]
        winner[better] = i

    result = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    for i in range(len(imgs_c)):
        result[winner == i] = samplers[i][winner == i]

    winner_u8 = ((winner + 1) * 30).astype(np.uint8)
    edges = cv2.Canny(winner_u8, 5, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    fzone = cv2.dilate(edges, kernel) > 0

    acc_f = np.zeros((OUT_H, OUT_W, 3), np.float64)
    wsum_f = np.zeros((OUT_H, OUT_W), np.float64)
    for i in range(len(imgs_c)):
        wm = np.where(masks[i] & fzone, np.cos(theta_maps[i]) ** 4, 0.).astype(np.float64)
        acc_f += samplers[i].astype(np.float64) * wm[:, :, None]
        wsum_f += wm

    blend = (acc_f / np.where(wsum_f > 0, wsum_f, 1)[:, :, None]).clip(0, 255).astype(np.uint8)
    result[fzone] = blend[fzone]

    covered = np.zeros((OUT_H, OUT_W), np.uint8)
    for m in masks:
        covered[m] = 1
    holes = (covered == 0).astype(np.uint8) * 255
    if holes.sum() > 0:
        log(f"  Inpainting {holes.sum() // 255:,} px…")
        result = cv2.inpaint(result, holes, 13, cv2.INPAINT_TELEA)

    result = result[:int(OUT_H * CROP_PCT)]
    log(f"  Rogné à {int(OUT_H * CROP_PCT)}px ({CROP_PCT * 100:.0f}%)")
    return result


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.current_directory = None
        self.output_path = None
        self.preview_ctk_image = None
        self.panorama_ctk_image = None
        self.file_checkboxes = {}
        self.running = False

        self.app_settings()
        self.withdraw()

        self.toplevel_window = TopWindow(self)
        self.toplevel_window.deiconify()
        self.toplevel_window.focus_force()

        self.geometry("1080x780")
        self.title("Panoramic View - Dashboard")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=1)

        self.file_list_frame = ctk.CTkScrollableFrame(self, label_text="Fichiers détectés")
        self.file_list_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Aucune image sélectionnée")
        self.preview_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.dropdown = ctk.CTkOptionMenu(
            self.control_frame,
            values=["Images cochées", "Toutes les images détectées"]
        )
        self.dropdown.pack(pady=(20, 10))

        self.action_button = ctk.CTkButton(
            self.control_frame,
            text="Lancer le traitement",
            fg_color="#27856A",
            hover_color="#2E9D7E",
            command=self.launch_processing
        )
        self.action_button.pack(pady=10)

        self.output_button = ctk.CTkButton(
            self.control_frame,
            text="Changer le fichier de sortie",
            command=self.choose_output
        )
        self.output_button.pack(pady=10)

        self.output_label = ctk.CTkLabel(
            self.control_frame,
            text="Sortie : aucune",
            justify="left",
            wraplength=520
        )
        self.output_label.pack(pady=(0, 10))

        self.progress_label = ctk.CTkLabel(self.control_frame, text="Progression : 0%")
        self.progress_label.pack(pady=(10, 4))

        self.progress_bar = ttk.Progressbar(self.control_frame, orient="horizontal", mode="determinate", length=340, maximum=100)
        self.progress_bar.pack(pady=(0, 10))

        self.log_textbox = ScrolledText(self.control_frame, width=75, height=10, wrap="word")
        self.log_textbox.pack(padx=20, pady=(10, 20), fill="both", expand=True)
        self.log_textbox.insert("end", "Logs du traitement...\n")
        self.log_textbox.configure(state="disabled")

        self.after(200, self.ask_output_on_startup)

    def app_settings(self):
        ctk.set_default_color_theme("gui/theme/sp_theme.json")
        ctk.set_appearance_mode("dark")
        scaling = get_scaling_ratio()
        ctk.set_window_scaling(scaling)
        ctk.set_widget_scaling(scaling)

    def append_log(self, msg):
        def _append():
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", str(msg) + "\n")
            self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")
        self.after(0, _append)

    def clear_logs(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def set_progress(self, value, label=None):
        value = max(0.0, min(100.0, float(value)))

        def _set():
            self.progress_bar["value"] = value
            text = f"Progression : {int(round(value))}%"
            if label:
                text = f"{label} ({int(round(value))}%)"
            self.progress_label.configure(text=text)
            self.update_idletasks()
        self.after(0, _set)

    def ask_output_on_startup(self):
        if self.output_path:
            return
        path = filedialog.asksaveasfilename(
            title="Choisir le fichier de sortie du panorama",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png")]
        )
        if path:
            self.output_path = path
            self.output_label.configure(text=f"Sortie : {path}")
            self.append_log(f"Fichier de sortie sélectionné : {path}")
        else:
            self.output_path = None
            self.output_label.configure(text="Sortie : aucune")
            self.append_log("Aucun fichier de sortie sélectionné.")

    def choose_output(self):
        path = filedialog.asksaveasfilename(
            title="Choisir le fichier de sortie",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png")]
        )
        if path:
            self.output_path = path
            self.output_label.configure(text=f"Sortie : {path}")
            self.append_log(f"Fichier de sortie sélectionné : {path}")

    def load_files(self, directory):
        self.current_directory = directory

        for widget in self.file_list_frame.winfo_children():
            widget.destroy()

        self.file_checkboxes = {}
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort()

        for file in files:
            container = ctk.CTkFrame(self.file_list_frame, fg_color="transparent")
            container.pack(fill="x", pady=2)

            lbl = ctk.CTkLabel(container, text=file, anchor="w", cursor="hand2")
            lbl.pack(side="left", fill="x", expand=True, padx=5)
            lbl.bind("<Button-1>", lambda e, f=file, d=directory: self.update_preview(d, f))

            cb = ctk.CTkCheckBox(
                container,
                text="",
                width=24,
                command=lambda f=file, d=directory: self.update_preview(d, f)
            )
            cb.pack(side="right", padx=5)
            self.file_checkboxes[file] = cb

    def update_preview(self, directory, filename):
        img_path = os.path.join(directory, filename)
        try:
            pil = Image.open(img_path)
            self.preview_ctk_image = ctk.CTkImage(light_image=pil, dark_image=pil, size=(400, 300))
            self.preview_label.configure(image=self.preview_ctk_image, text="")
        except Exception as e:
            self.preview_label.configure(text=f"Erreur : {e}", image=None)

    def get_selected_paths(self):
        if not self.current_directory:
            return []

        mode = self.dropdown.get()
        if mode == "Toutes les images détectées":
            names = list(self.file_checkboxes.keys())
        else:
            names = [name for name, cb in self.file_checkboxes.items() if cb.get() == 1]

        names.sort()
        return [os.path.join(self.current_directory, name) for name in names[:8]]

    def launch_processing(self):
        if self.running:
            return

        paths = self.get_selected_paths()
        if len(paths) < 2:
            messagebox.showerror("Erreur", "Sélectionne au moins 2 images.")
            return

        if not self.output_path:
            self.ask_output_on_startup()
            if not self.output_path:
                messagebox.showerror("Erreur", "Choisis un fichier de sortie avant de lancer le traitement.")
                return

        self.running = True
        self.clear_logs()
        self.append_log("=" * 55)
        self.append_log("Traitement du panorama")
        self.append_log("=" * 55)
        self.set_progress(0, "Initialisation")
        self.action_button.configure(state="disabled", text="Traitement…")

        threading.Thread(target=self._worker, args=(paths,), daemon=True).start()

    def _worker(self, paths):
        try:
            def log(msg):
                self.append_log(msg)

            self.set_progress(5, "Chargement des images")

            raw = []
            total_paths = max(1, len(paths))
            log("Chargement :")
            for idx, path in enumerate(paths, start=1):
                img = safe_imread(path)
                if img is None:
                    log(f"  ✗ {os.path.basename(path)} introuvable")
                else:
                    log(f"  ✓ {os.path.basename(path)}  ({img.shape[1]}×{img.shape[0]})")
                    raw.append(img)
                self.set_progress(5 + (idx / total_paths) * 15, "Chargement des images")

            if len(raw) < 2:
                raise RuntimeError("Moins de 2 images.")

            self.set_progress(25, "Préparation du matching")
            imgs_match = [resize_keep(img, MATCH_SIZE) for img in raw]

            self.set_progress(35, "Bundle adjustment")
            f_ba, quats = bundle_adjustment(imgs_match, raw, log=log)

            self.set_progress(75, "Projection sphérique")
            log("\n[3/3] Sauvegarde…")
            panorama = project_sphere(raw, f_ba, quats, MATCH_SIZE, log=log)

            self.set_progress(92, "Écriture du fichier")
            safe_imwrite(self.output_path, panorama, quality=94)
            log(f"✓  {self.output_path}  ({panorama.shape[1]}×{panorama.shape[0]} px)")

            self.set_progress(100, "Terminé")
            self.after(0, lambda: self.show_panorama_preview(panorama))
            self.after(0, lambda: messagebox.showinfo("Succès", f"Panorama créé :\n{self.output_path}"))
        except Exception as e:
            tb = traceback.format_exc()
            self.append_log("\n[ERREUR]")
            self.append_log(str(e))
            self.append_log(tb)
            self.set_progress(0, "Erreur")
            self.after(0, lambda: messagebox.showerror("Erreur", f"{e}\n\n{tb}"))
        finally:
            self.running = False
            self.after(0, lambda: self.action_button.configure(state="normal", text="Lancer le traitement"))

    def show_panorama_preview(self, panorama_bgr):
        rgb = cv2.cvtColor(panorama_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((400, 300))
        self.panorama_ctk_image = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self.preview_label.configure(image=self.panorama_ctk_image, text="")


def get_scaling_ratio():
    root = tkinter.Tk()
    root.update_idletasks()
    root.withdraw()
    current_dpi = root.winfo_fpixels('1i')
    scaling = current_dpi / 96.0
    root.destroy()
    return scaling
