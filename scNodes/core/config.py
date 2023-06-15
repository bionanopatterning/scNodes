from joblib import cpu_count
import traceback
import dill as pickle
import os
from datetime import datetime

frozen = False
root = ""
app_name = "scNodes"
version = "1.1.16"
license = "GNU GPL v3"
logpath = "scNodes.log"
filetype_project = ".scnp"
filetype_node_setup = ".scn"
filetype_scene = ".scnscene"
filetype_segmentation = ".scns"
filetype_traindata = ".scnt"
filetype_semodel = ".scnm"

controls = [
    ("0/H/V", "Toggle frame visibility"),
    ("1-6", "Change blending mode"),
    ("-/+", "De-/increase frame alpha"),
    ("A", "Set autocontrast. +SHIFT: higher contrast, +CTRL: even higher contrast"),
    ("C", "Toggle clamp mode (clamp / discard / discard min)"),
    ("E/P", "Enable/disable particle picking"),
    ("I", "Toggle interpolation mode. +SHIFT: invert contrast"),
    ("L", "Toggle current frame edit lock"),
    ("M", "(De)activate measure tool"),
    ("S", "Toggle snapping"),
    ("[", "Toggle particle picking mode (single particle / filament)"),
    ("Arrows", "Move frame. +SHIFT: move fast, +CTRL: move slow"),
    ("Escape", "Deactivate measure tool / particle picking"),
    ("Shift + scroll", "Zoom in/out"),
    ("Delete", "Delete frame"),
    ("Page up/down", "Move frame up/down in render stack. +SHIFT: move to top/bottom"),
    ("Spacebar", "Focus camera on currently selected frame"),
]

nodes = list()
active_node = None
focused_node = None
any_change = False  # top-level change flag
next_active_node = None
node_move_requested = [0, 0]
camera_move_requested = [0, 0]

window_width = 1100
window_height = 700
iv_window_width = 600
iv_window_height = 700
cursor_pos = [0, 0]
profiling = False

error_msg = None
error_new = True
error_obj = None
error_logged = False
error_window_active = False

active_connector = None
active_connector_parent_node = None
active_connector_hover_pos = [0, 0]
connector_released = False
connector_delete_requested = False
ne_dropped_files = None
n_cpus_max = cpu_count()
n_cpus = n_cpus_max
batch_size = n_cpus_max * 3

image_viewer = None
node_editor = None
correlation_editor = None
segmentation_editor = None
ce_tool_menu_names = {'Transform': True, 'Visuals': True, 'Export': True, 'Measure': True, 'Plugins': True, 'Particle picking': True}
node_editor_relink = False
correlation_editor_relink = False
pickle_temp = dict()

editors = ["Node Editor", "Correlation Editor", "Segmentation Editor"]
active_editor = 0  # 0 for node editor, 1 for correlation
ce_frames = list()
ce_active_frame = None
ce_clear_colour = (1.0, 1.0, 1.0, 1.0)
ce_default_pixel_size = 64.0
ce_flip_on_load = True
ce_selected_position = [0, 0]
ce_va_subdivision = 8

se_frames = list()
se_active_frame = None
se_models = list()
se_active_model = None
se_path = "..."


def set_active_node(node, keep_active=False):
    global focused_node, active_node, next_active_node
    """
    :param node: Node type object or None.
    :param keep_active: bool. If True, the node is not just set as active_node, but also as the focused_node - this means that the node is kept in focus until it is manually unfocused by the user (in node right click context menu).
    :return: False if no change; i.e. when input node is already the active node. True otherwise.
    """
    if keep_active:
        focused_node = node
        next_active_node = node
        return True
    elif focused_node is None:
        if active_node is not None:
            if active_node == node:
                return False
        if node is None:
            active_node = None
            return True
        else:
            next_active_node = node
            return True
    else:
        return False


def set_error(error_object, error_message):
    global error_msg, error_obj, error_new, error_logged
    error_msg = error_message + "\n\n"
    error_msg += "".join(traceback.TracebackException.from_exception(error_object).format())
    print(error_msg)
    error_obj = error_object
    error_new = True
    error_logged = False


def save_scene(filename):
    if not filetype_scene in filename:
        filename += filetype_scene
    with open(filename, 'wb') as pickle_file:
        pickle.dump(ce_frames, pickle_file)


def load_scene(filename, append=False):
    global ce_frames, correlation_editor_relink
    try:
        with open(filename, 'rb') as pickle_file:
            imported_ce_frames = pickle.load(pickle_file)
            for cef in imported_ce_frames:
                if not hasattr(cef, "sum_slices"):
                    cef.sum_slices = 1
                if not hasattr(cef, "current_sum_slices"):
                    cef.current_sum_slices = 1
                if not hasattr(cef, "locked"):
                    cef.locked = False
            if append:
                ce_frames += imported_ce_frames
            else:
                ce_frames = imported_ce_frames
            correlation_editor_relink = True
    except Exception as e:
        set_error(e, "Error loading scene")


def write_to_log(text):
    with open(os.path.join(root, logpath), "a") as f:
        f.write("\n\n ____________________ \n\n")
        f.write(text)


def start_log():
    if os.path.join(root, logpath):
        os.path.join(root, logpath)
    with open(os.path.join(root, logpath), "w") as f:
        f.write(app_name+" version "+version+" "+license+"\n"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))


COLOUR_TEST_A = (1.0, 0.0, 1.0, 1.0)
COLOUR_TEST_B = (0.0, 1.0, 1.0, 1.0)
COLOUR_TEST_C = (1.0, 1.0, 0.0, 1.0)
COLOUR_TEST_D  = (1.0, 1.0, 1.0, 1.0)

COLOUR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 0.94)
COLOUR_PANEL_BACKGROUND = (0.94, 0.94, 0.94, 0.94)
COLOUR_TITLE_BACKGROUND = (0.87, 0.87, 0.83, 0.96)
COLOUR_TITLE_BACKGROUND_LIGHT = (0.96, 0.96, 0.93, 0.93)
COLOUR_FRAME_BACKGROUND = (0.87, 0.87, 0.83, 0.96)
COLOUR_FRAME_ACTIVE = (0.91, 0.91, 0.86, 0.94)
COLOUR_FRAME_DARK = (0.83, 0.83, 0.76, 0.94)
COLOUR_FRAME_EXTRA_DARK = (0.76, 0.76, 0.71, 0.94)
COLOUR_MAIN_MENU_BAR = (0.882, 0.882, 0.882, 0.94)
COLOUR_MAIN_MENU_BAR_TEXT = (0.0, 0.0, 0.0, 0.94)
COLOUR_MAIN_MENU_BAR_HILIGHT = (0.96, 0.95, 0.92, 0.94)
COLOUR_MENU_WINDOW_BACKGROUND = (0.96, 0.96, 0.96, 0.94)
COLOUR_DROP_TARGET = COLOUR_FRAME_DARK
COLOUR_HEADER = COLOUR_FRAME_DARK
COLOUR_HEADER_ACTIVE = COLOUR_FRAME_ACTIVE
COLOUR_HEADER_HOVERED = COLOUR_FRAME_EXTRA_DARK
COLOUR_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_TEXT_ACTIVE = (0.0, 0.0, 0.2, 1.0)
COLOUR_TEXT_DISABLED = (0.7, 0.7, 0.7, 1.0)
COLOUR_TEXT_FADE = COLOUR_FRAME_EXTRA_DARK
WINDOW_ROUNDING = 5.0
CONTEXT_MENU_SIZE = (200, 98)
ERROR_WINDOW_HEIGHT = 80
COLOUR_ERROR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 0.94)
COLOUR_ERROR_WINDOW_HEADER = (0.87, 0.87, 0.83, 0.96)
COLOUR_ERROR_WINDOW_HEADER_NEW = (0.87, 0.87, 0.83, 0.96)
COLOUR_ERROR_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_CM_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_CM_OPTION_HOVERED = (1.0, 1.0, 1.0, 1.0)
COLOUR_TRANSPARENT = (1.0, 1.0, 1.0, 0.0)
COLOUR_FRAME_BACKGROUND_BLUE = (0.76, 0.76, 0.83, 1.0)
COLOUR_POSITIVE = (0.1, 0.8, 0.1, 1.0)
COLOUR_NEGATIVE = (0.8, 0.1, 0.1, 1.0)
COLOUR_NEUTRAL = (0.6, 0.6, 0.6, 1.0)
COLOUR_NEUTRAL_LIGHT = (0.8, 0.8, 0.8, 1.0)

TOOLTIP_APPEAR_DELAY = 1.0
TOOLTIP_HOVERED_TIMER = 0.0
TOOLTIP_HOVERED_START_TIME = 0.0

CE_WIDGET_ROUNDING = 50.0