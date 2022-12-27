from joblib import cpu_count
import traceback
# This file defines variables that can be accessed globally.
# Before re-structuring the code, the Node class would directly access and change NodeEditor class static variables.
# That messed up inheritance. Instead, all of the variables that were accessed in this way are now defined in this file.

nodes = list()
active_node = None
focused_node = None
any_change = False  # top-level change flag
next_active_node = None

node_move_requested = [0, 0]
camera_move_requested = [0, 0]

window_width = 1920
window_height = 1080

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

## CONFIG HAS NO ATTRIB IMG VIEWER 221223

pickle_temp = dict()

## 221221 correlation editor vars & related
active_editor = 0  # 0 for node editor, 1 for correlation

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