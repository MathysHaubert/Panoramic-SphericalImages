import customtkinter as ctk
from OpenGL.GL import *
from OpenGL.GLU import *
from pyopengltk import OpenGLFrame
from PIL import Image
import numpy as np

class PanoramaWindow(ctk.CTkToplevel):
    def __init__(self, image_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Visualisation 360°")
        self.geometry("900x600")

        self.viewer = PanoramaEngine(self, image_path)
        self.viewer.pack(expand=True, fill="both")

        self.label = ctk.CTkLabel(self, text="Cliquez et glissez pour explorer", fg_color="transparent")
        self.label.place(relx=0.5, rely=0.05, anchor="center")


class PanoramaEngine(OpenGLFrame):
    def __init__(self, master, image_path, **kwargs):
        super().__init__(master, **kwargs)
        self.image_path = image_path
        self.yaw = 0
        self.pitch = 0
        self.last_x = 0
        self.last_y = 0
        self.tex_id = None

    def initgl(self):
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        img = Image.open(self.image_path).transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(img.convert("RGB"), np.uint8)

        self.tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

        self.bind("<B1-Motion>", self.mouse_move)
        self.bind("<Button-1>", self.mouse_press)
        self.tkExpose(None)

    def redraw(self):
        if self.tex_id is None:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(75, (self.winfo_width() / self.winfo_height()), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.yaw, 0, 0, 1)

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        quad = gluNewQuadric()
        gluQuadricTexture(quad, GL_TRUE)
        gluQuadricOrientation(quad, GLU_INSIDE)
        gluSphere(quad, 10, 64, 64)
        gluDeleteQuadric(quad)

    def mouse_press(self, event):
        self.last_x, self.last_y = event.x, event.y

    def mouse_move(self, event):
        dx = event.x - self.last_x
        dy = event.y - self.last_y

        self.yaw += dx * 0.2
        self.pitch += dy * 0.2
        self.pitch = max(-180.0, min(180.0, self.pitch))

        self.last_x, self.last_y = event.x, event.y
        self.tkExpose(None)