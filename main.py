import settings
import glfw
import OpenGL.GL
from window import *
from nodeeditor import *
from imageviewer import *
from reconstruction import *
from util import *

if __name__ == "__main__":
    if not glfw.init():
        raise Exception("Could not initialize GLFW library")


    ne_window = Window(settings.ne_window_width, settings.ne_window_height, settings.ne_window_title)
    node_editor = NodeEditor(ne_window)
    cfg.node_editor = node_editor
    node_editor.delete_temporary_files()

    iv_window = Window(settings.iv_window_width, settings.iv_window_height, settings.iv_window_title)
    image_viewer = ImageViewer(iv_window, node_editor.get_font_atlas_ptr())
    cfg.image_viewer = image_viewer

    try:
        while not glfw.window_should_close(ne_window.glfw_window):
            if not (ne_window.focused or iv_window.focused):
                glfw.poll_events()


            image_viewer.on_update()
            image_viewer.end_frame() # imageviewer was AFTER nodeedotir

            node_editor.on_update()
            node_editor.end_frame()
    finally:
        node_editor.delete_temporary_files()



