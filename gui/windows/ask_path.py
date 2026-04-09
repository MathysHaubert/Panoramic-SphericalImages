import customtkinter as ctk
from tkinter import filedialog


class TopWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("500x250")
        self.title("Configuration")

        self.resizable(False, False)

        self.label = ctk.CTkLabel(self, text="Dossier des images panoramiques", font=("Roboto", 15, "bold"))
        self.label.pack(pady=(20, 10))

        self.entry_path = ctk.CTkEntry(self, width=400, placeholder_text="Cliquez sur 'Parcourir'...")
        self.entry_path.pack(pady=10, padx=20)

        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=20)

        self.btn_browse = ctk.CTkButton(self.btn_frame, text="Parcourir", width=120, command=self.browse)
        self.btn_browse.pack(side="left", padx=10)

        self.btn_confirm = ctk.CTkButton(self.btn_frame, text="Valider", width=120,
                                         fg_color="#27856A", hover_color="#2E9D7E",
                                         command=self.validate)
        self.btn_confirm.pack(side="left", padx=10)

    def browse(self):
        path = filedialog.askdirectory(title="Choisir le dossier des images")
        if path:
            self.entry_path.delete(0, "end")
            self.entry_path.insert(0, path)

    def validate(self):
        path = self.entry_path.get()
        if path:
            self.master.load_files(path)
            self.master.deiconify()
            self.destroy()
