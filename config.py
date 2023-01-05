from joblib import cpu_count
import traceback
import dill as pickle
# This file defines variables that can be accessed globally.
# Before re-structuring the code, the Node class would directly access and change NodeEditor class static variables.
# That messed up inheritance. Instead, all of the variables that were accessed in this way are now defined in this file.

app_name = "srNodes"

nodes = list()
active_node = None
focused_node = None
any_change = False  # top-level change flag
next_active_node = None
node_move_requested = [0, 0]
camera_move_requested = [0, 0]

window_width = 1100
window_height = 700

profiling = False

error_msg = None
error_new = True
error_obj = None

active_connector = None
active_connector_parent_node = None
active_connector_hover_pos = [0, 0]
connector_released = False
connector_delete_requested = False

n_cpus_max = cpu_count()
n_cpus = n_cpus_max
batch_size = n_cpus_max

image_viewer = None
node_editor = None
correlation_editor = None
node_editor_relink = False
correlation_editor_relink = False
pickle_temp = dict()


## 221221 correlation editor vars & related
active_editor = 0  # 0 for node editor, 1 for correlation
ce_frames = list()
ce_clear_colour = (1.0,1.0,1.0,1.0)
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
    global error_msg, error_obj, error_new
    error_msg = error_message + "\n\n"
    error_msg += "".join(traceback.TracebackException.from_exception(error_object).format())
    print(error_msg)
    error_obj = error_object
    error_new = True

def save_project(filename):
    if filename[-5:] == '.srnp':
        filename = filename[:-5]
    with open(filename+'.srnp', 'wb') as pickle_file:
        pickle.dump([nodes, ce_frames], pickle_file)

def load_project(filename):
    global nodes, ce_frames, correlation_editor_relink, node_editor_relink
    try:
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            nodes = data[0]
            ce_frames = data[1]
            node_editor_relink = True
            correlation_editor_relink = True
    except Exception as e:
        set_error(e, "Error loading project")

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