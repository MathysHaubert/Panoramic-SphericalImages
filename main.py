import os
os.environ['PYOPENGL_PLATFORM'] = 'x11'
from gui.app import App

def start():

    # Launch GUI
    app = App()
    app.mainloop()


if __name__ == '__main__':
    start()

