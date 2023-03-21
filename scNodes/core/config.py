from joblib import cpu_count
import traceback
import dill as pickle
import os
from datetime import datetime
# This file defines variables that can be accessed globally.

# TO DO list
# todo: fix error where bakestack freezes baking a second time. 230309 - fixed, maybe?
# todo: when flipping a frame in CE, flip children as well.
# todo: input - press '0' (zero) to hide/show frame
# todo: input - press "SHIFT" and "+" or "-" to increase/decrease active frame Alpha by 0.1
# todo: in the View tab in the CE, set the imgui frame padding to 2
# todo: in the View tab in the CE: hide the tool window when all views set to False
# todo: fix the Register grayscale plugin
# todo: input - press "I" to toffle interpolation
# todo: fix mrc file clemframes sometimes not updating despite slicer being changed and frame being loaded (the last thing I presume, since there is a slight loading delay upon changing the slicer value). Reason appears to be that MIPMAP IS ONLY SET WHEN INTERP MODE CHANGES. NO MIPMAPPING FOR MRC FILES!
# todo: add alpha wobbler in right-click menu for the alpha slider
# todo: fix BLUR FRAME plugin's original frame still being visible in the scene, but not in the object list
# todo: add feature to CE: left-click and hold on a (pile of) frames opens a popup that lists all the files under the cursor. Helps to select files hidden by a file higher up in the stack - like in Fusion360.
frozen = False
root = ""
app_name = "scNodes"
version = "1.1.8"
license = "GNU GPL v3"
logpath = "scNodes.log"
filetype_project = ".scnp"
filetype_node_setup = ".scn"
filetype_scene = ".scnscene"

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

profiling = False

error_msg = None
error_new = True
error_obj = None
error_logged = False

active_connector = None
active_connector_parent_node = None
active_connector_hover_pos = [0, 0]
connector_released = False
connector_delete_requested = False

n_cpus_max = cpu_count()
n_cpus = n_cpus_max
batch_size = n_cpus_max * 3

image_viewer = None
node_editor = None
correlation_editor = None
ce_tool_menu_names = {'Transform': True, 'Visuals': True, 'Export': True, 'Measure': True, 'Plugins': True, 'Particle picking': True}
node_editor_relink = False
correlation_editor_relink = False
pickle_temp = dict()


## 221221 correlation editor vars & related
active_editor = 0  # 0 for node editor, 1 for correlation
ce_frames = list()
ce_active_frame = None
ce_clear_colour = (1.0, 1.0, 1.0, 1.0)
ce_default_pixel_size = 64.0
ce_flip_on_load = False
ce_selected_position = [0, 0]
ce_va_subdivision = 8

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

def load_scene(filename):
    global ce_frames, correlation_editor_relink
    try:
        with open(filename, 'rb') as pickle_file:
            ce_frames = pickle.load(pickle_file)
            correlation_editor_relink = True
    except Exception as e:
        set_error(e, "Error loading scene")


def write_to_log(text):
    with open(root+logpath, "a") as f:
        f.write("\n\n ____________________ \n\n")
        f.write(text)

def start_log():
    if os.path.exists(root+logpath):
        os.remove(root+logpath)
    with open(root+logpath, "a") as f:
        f.write(app_name+" version "+version+" "+license+"\n"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

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
COLOUR_TEXT_FADE = COLOUR_FRAME_EXTRA_DARK
WINDOW_ROUNDING = 5.0
CONTEXT_MENU_SIZE = (200, 98)
ERROR_WINDOW_HEIGHT = 80
COLOUR_ERROR_WINDOW_BACKGROUND = (0.84, 0.84, 0.84, 1.0)
COLOUR_ERROR_WINDOW_HEADER = (0.7, 0.7, 0.7, 1.0)
COLOUR_ERROR_WINDOW_HEADER_NEW = (0.35, 0.35, 0.35, 1.0)
COLOUR_ERROR_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_CM_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
COLOUR_CM_OPTION_HOVERED = (1.0, 1.0, 1.0, 1.0)

TOOLTIP_APPEAR_DELAY = 1.0
TOOLTIP_HOVERED_TIMER = 0.0
TOOLTIP_HOVERED_START_TIME = 0.0

CE_WIDGET_ROUNDING = 50.0
