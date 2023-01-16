from src.ceplugin import *


def create():
    return CustomPlugin()


class CustomPlugin(CEPlugin):
    title = "Custom plugin template"
    description = "The description to be shown when the '?' icon is hovered goes here"
    enabled = False  # default is True. Set to False in order not to load the plugin upon booting the software.
    FLAG_SHOW_LOCATION_PICKER = True  # default is False. Set to True to enable a location marker for this plugin (as in the Register Grayscale node)

    def __init__(self):
        ## variables for settings can be initialized here.
        pass

    def render(self):
        # body of the node goes here.
        # see ceplugin.py for a list of predefined widgets that may be useful.
        # these can be called as in the following examples:

        self.widget_show_active_frame_title() # text field showing the title of the currently selected frame

        _selection_changed, frame = self.widget_select_frame_any("Select frame", 0)  # drop-down menu with all frames currently in the scene available for selection. _selection_changed is a bool flagging whether the selected frame was changed
        _selection_changed, frame = self.widget_select_frame_no_rgb("Select frame", 0)  # drop-down menu with all grayscale frames currently in the scene available for selection
        _selection_changed, frame = self.widget_select_frame_rgb("Select frame", 0)  # drop-down menu with all rgb frames currently in the scene available for selection

        self.widget_selected_position()  # shows the current coordinates of the location picker

        if self.widget_centred_button("Click me"):  # adds a button that is horizontally centred in the window
            self.process_frame(frame)
        self.tooltip("This text shows when the button is hovered.")

    @staticmethod
    def process_frame(frame):
        pass  # this function is called when the user presses the 'Click me' button that was defined above. Functionality of the plugin may go here (or simply within the if clause in the render function)
