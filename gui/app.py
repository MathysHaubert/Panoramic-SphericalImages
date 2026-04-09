import customtkinter as ctk
import os
import tkinter
from PIL import Image

from gui.widgets.sphere import PanoramaWindow
from gui.windows.ask_path import TopWindow


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.app_settings()
        self.withdraw()

        self.toplevel_window = TopWindow(self)

        self.toplevel_window.deiconify()
        self.toplevel_window.focus_force()
        self.geometry("1000x700")
        self.title("Panoramic View - Dashboard")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=1)

        self.file_list_frame = ctk.CTkScrollableFrame(self, label_text="Fichiers détectés")
        self.file_list_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.file_checkboxes = {}

        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Aucune image sélectionnée")
        self.preview_label.pack(expand=True, fill="both")

        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.dropdown = ctk.CTkOptionMenu(self.control_frame, values=["Option 1", "Option 2", "Option 3"])
        self.dropdown.pack(pady=(20, 10))

        self.action_button = ctk.CTkButton(self.control_frame, text="Lancer le traitement",
                                           fg_color="#27856A", hover_color="#2E9D7E",
                                           command=self.test_panorama)
        self.action_button.pack(pady=10)

    def load_files(self, directory):
        """Appelée une fois le chemin validé dans TopWindow"""

        for widget in self.file_list_frame.winfo_children():
            widget.destroy()

        files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for file in files:
            container = ctk.CTkFrame(self.file_list_frame, fg_color="transparent")
            container.pack(fill="x", pady=2)

            lbl = ctk.CTkLabel(container, text=file, anchor="w", cursor="hand2")
            lbl.pack(side="left", fill="x", expand=True, padx=5)
            lbl.bind("<Button-1>", lambda e, f=file, d=directory: self.update_preview(d, f))

            cb = ctk.CTkCheckBox(container, text="", width=24)
            cb.pack(side="right", padx=5)
            self.file_checkboxes[file] = cb

    def update_preview(self, directory, filename):
        """Affiche l'image sélectionnée"""
        img_path = os.path.join(directory, filename)
        try:

            my_image = ctk.CTkImage(light_image=Image.open(img_path),
                                    dark_image=Image.open(img_path),
                                    size=(400, 300))
            self.preview_label.configure(image=my_image, text="")
        except Exception as e:
            self.preview_label.configure(text=f"Erreur : {e}")

    def app_settings(self):
        ctk.set_default_color_theme("gui/theme/sp_theme.json")
        ctk.set_appearance_mode("dark")
        scaling = get_scaling_ratio()
        ctk.set_window_scaling(scaling)
        ctk.set_widget_scaling(scaling)

    def on_processing_finished(self, final_image_path):
        # Ouvre la fenêtre de visualisation
        view_window = PanoramaWindow(final_image_path)
        view_window.focus()

    def test_panorama(self):
        path_to_test = r"/home/mathys/PycharmProjects/Panoramic-SphericalImages/images/Example/FinalResult.jpg"

        if os.path.exists(path_to_test):
            print("Test du panorama lancé...")
            self.on_processing_finished(path_to_test)
        else:
            print(f"Erreur : Le fichier est introuvable à l'adresse {path_to_test}")


def get_scaling_ratio():
    root = tkinter.Tk()
    root.update_idletasks()
    root.withdraw()
    current_dpi = root.winfo_fpixels('1i')
    scaling = current_dpi / 96.0
    root.destroy()
    return scaling
