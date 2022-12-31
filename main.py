import settings
import glfw
import OpenGL.GL
from window import *
from nodeeditor import *
from correlation_editor import *
from imageviewer import *
from reconstruction import *
from util import *
from joblib.externals.loky import set_loky_pickler
set_loky_pickler("cloudpickle")

if __name__ == "__main__":
    if not glfw.init():
        raise Exception("Could not initialize GLFW library!")


    # Init the main window, its imgui context, and a glfw rendering impl.
    main_window = Window(settings.ne_window_width, settings.ne_window_height, settings.ne_window_title)
    main_window.set_callbacks()
    main_window_imgui_context = imgui.create_context()
    main_window_imgui_glfw_implementation = GlfwRenderer(main_window.glfw_window)
    main_window.set_mouse_callbacks()
    node_editor = NodeEditor(main_window, main_window_imgui_context, main_window_imgui_glfw_implementation)
    correlation_editor = CorrelationEditor(main_window, main_window_imgui_context, main_window_imgui_glfw_implementation)

    image_viewer_window = Window(settings.iv_window_width, settings.iv_window_height, settings.iv_window_title)
    image_viewer_window_imgui_context = imgui.create_context(main_window_imgui_glfw_implementation.io.fonts)
    image_viewer_window_glfw_implementation = GlfwRenderer(image_viewer_window.glfw_window)
    image_viewer = ImageViewer(image_viewer_window, image_viewer_window_imgui_context, image_viewer_window_glfw_implementation)
    node_editor.delete_temporary_files()
    cfg.node_editor = node_editor
    cfg.correlation_editor = correlation_editor
    cfg.image_viewer = image_viewer


    try:
        while not glfw.window_should_close(main_window.glfw_window):
            if not (main_window.focused or image_viewer_window.focused):
                glfw.poll_events()

            if cfg.active_editor == 0:
                node_editor.on_update()
                node_editor.end_frame()
            else:
                correlation_editor.on_update()
                correlation_editor.end_frame()

            image_viewer.on_update()
            image_viewer.end_frame()

    finally:
        node_editor.delete_temporary_files()



