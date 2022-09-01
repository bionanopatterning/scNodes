import glfw
from OpenGL.GL import *
import config as cfg


class Window:
    def __init__(self, width, height, title, main_window = False):
        self.width = width
        self.height = height
        self.title = title
        self.clear_color = (0.0, 0.0, 0.0, 1.0)
        self.glfw_window = glfw.create_window(self.width, self.height, self.title, None, None)
        self.focused = True
        self.main_window = main_window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, OpenGL.GL.GL_TRUE)
        glfw.make_context_current(self.glfw_window)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        if not self.glfw_window:
            glfw.terminate()
            raise Exception("Could not initialize GLFW window.")

        # input vars
        self.cursor_pos = [0, 0]
        self.cursor_delta = [0, 0]
        self.scroll_delta = [0, 0]
        self.mouse_event = MouseButtonEvent(None, None, None)
        self.key_event = KeyEvent(None, None, None)


        # Change flags
        self.window_size_changed = False

        # default callbacks
        glfw.set_window_focus_callback(self.glfw_window, self.window_focus_callback)

    def set_callbacks(self):
        glfw.set_window_size_callback(self.glfw_window, self.size_changed_callback)
        glfw.set_key_callback(self.glfw_window, self.key_callback)
        glfw.set_mouse_button_callback(self.glfw_window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.glfw_window, self.scroll_callback)

    def set_mouse_callbacks(self):
        glfw.set_mouse_button_callback(self.glfw_window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.glfw_window, self.scroll_callback)

    def on_update(self):
        self.scroll_delta = [0.0, 0.0]
        glClearColor(*self.clear_color)
        glClear(GL_COLOR_BUFFER_BIT)
        _current_cursor_pos = list(glfw.get_cursor_pos(self.glfw_window))
        self.cursor_delta = [_current_cursor_pos[0] - self.cursor_pos[0], _current_cursor_pos[1] - self.cursor_pos[1]]
        self.cursor_pos = _current_cursor_pos
        if self.focused:
            glfw.poll_events()

    def make_current(self):
        glfw.make_context_current(self.glfw_window)

    def set_full_viewport(self):
        glViewport(0, 0, self.width, self.height)

    def end_frame(self):
        glfw.swap_buffers(self.glfw_window)

    def size_changed_callback(self, window, width, height):
        self.width = max([width, 1])
        self.height = max([height, 1])

        glViewport(0, 0, self.width, self.height)
        self.window_size_changed = True

    def scroll_callback(self, _, dx, dy):
        self.scroll_delta = [dx, dy]

    def mouse_button_callback(self, _, button, action, mods):
        self.mouse_event = MouseButtonEvent(button, action, mods)

    def key_callback(self, window, button, scancode, action, mods):
        self.key_event = KeyEvent(button, action, mods)

    def window_focus_callback(self, window, focused):
        if focused:
            self.focused = True
        else:
            self.focused = False

    def get_mouse_button(self, button):
        return glfw.get_mouse_button(self.glfw_window, button)

    def get_key_event(self, key, action, mods = 0, pop_event = True):
        if self.key_event:
            if self.key_event.check(key, action, mods):
                if pop_event:
                    self.key_event = None
                return True
        return False

    def get_key(self, key):
        return glfw.get_key(self.glfw_window, key)

    def get_mouse_event(self, button, action, mods=0, pop_event=True):
        if self.mouse_event:
            if self.mouse_event.check(button, action, mods):
                if pop_event:
                    self.mouse_event = None
                return True
        return False


class KeyEvent:
    def __init__(self, key, action, mods):
        self.key = key
        self.action = action
        self.mods = mods

    def check(self, requested_key, requested_action, requested_mods):
        if self.key == requested_key:
            if self.action == requested_action:
                if self.mods == requested_mods:
                    return True
        return False

    def __str__(self):
        return f"key {self.key} action {self.action} with modifiers {self.mods}"

class MouseButtonEvent:
    def __init__(self, button, action, mods):
        self.button = button
        self.action = action
        self.mods = mods

    def check(self, requested_button, requested_action, requested_mods):
        #print(self.button, requested_button, self.action, requested_action, self.mods, requested_mods)
        if self.button == requested_button:
            if self.action == requested_action:
                if self.mods == requested_mods:
                    return True
        return False

