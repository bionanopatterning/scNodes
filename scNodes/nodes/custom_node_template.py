from scNodes.node import *

"""
This file can be used as a template to build a custom node. See the user manual for more information.
For a node to be properly registered by the software, the .py file that defines the node must include:
i) a function 'create' that returns an object of type Node
ii) a class that defines said Node object. A new node is easily set up by making a new class, e.g. 'CustomNode,' that inherits from the Node base class.
"""


def create():
    return CustomNode()


class CustomNode(Node):
    title = "Example node"  # The title is the name by which the node appears in the editor.
    group = ["Tutorial", "Custom nodes"]  # In the 'add node' right-click context menu, nodes are listed in groupes. A node can be in multiple groups as in this example. Any group name can be used here - also groups that are not in the default software.
    colour = (1.0, 0.0, 1.0, 1.0)  # this node will be magenta.
    sortid = 0  # upon initializing software, nodes are loaded in order of increasing sortid value. This determines the order with which they are presented in the editor right click context menu.
    enabled = False

    def __init__(self):
        super().__init__()  # In this line the init function of the Node parent class is called (can be found in node.py) - you can ignore it but it must be called.
        self.size = 300  # Set the horizontal size of the node.

        # defining in- and output attributes.
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)  # In this example the node has two 'connectable attributes': a dataset input and output.
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)
        # below we define some variables that are used for the example imgui features.
        self.noise_stdev = 1.0
        self.n_button_clicks = 0
        self.stopwatch = False
        self.stopwatch_start = 0
        self.stopwatch_now = 0

    def render(self):
        if super().render_start(): # as in the above __init__ function, the render function must by calling the base class' render_start() function, and end with a matching render_end() - see below.
            self.dataset_in.render_start()  # calling a ConnectableAttribute's render_start() and render_end() handles the rendering and connection logic for that attribute.
            self.dataset_out.render_start()
            self.dataset_in.render_end()
            self.dataset_out.render_end() # callin start first for both attributes and end afterwards makes the attributes appear on the same line / at the same height.

            imgui.spacing()  # add a small vertical whitespace
            imgui.separator()  # draw a line separating the above connectors from the rest of the body of the node. purely visual.
            imgui.spacing()

            # THE BELOW SECTION SHOWCASES SOME COMMON IMGUI FEATURES THAT CAN BE USED. For a detailed guide refer to the documentation of pyimgui: https://pyimgui.readthedocs.io/en/latest/
            imgui.text("This custom node generates noise.")
            imgui.set_next_item_width(150)
            _c, self.noise_stdev = imgui.slider_float("Noise stdev", self.noise_stdev, 0.0, 10.0)
            self.any_change = self.any_change or _c  # Node.any_change is one of various flags; boolean member variables of the Node base class that trigger certain behaviours. See the user manual for more info. This particular flag notifies the image viewer that a change has been made to the settings, and triggers the image viewer to request an updated output image.

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            button_width = 80
            imgui.new_line()
            remaining_window_content_width = imgui.get_window_content_region_width()
            imgui.same_line(position=remaining_window_content_width / 2 - button_width / 2)
            button_clicked = imgui.button("Hover me", width = button_width, height = 20)
            self.tooltip("now click to increase the counter.")
            if button_clicked:
                self.n_button_clicks += 1
            imgui.text(f"The button was clicked {self.n_button_clicks} times.")

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _changed, self.colour = imgui.color_edit4("Node colour", *self.colour, imgui.COLOR_EDIT_NO_INPUTS)

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        _changed, self.stopwatch = imgui.checkbox("Stopwatch on", self.stopwatch)
        if _changed:
            self.stopwatch_start = datetime.datetime.now()
            self.stopwatch_now = datetime.datetime.now()
        if self.stopwatch:
            delta_time = self.stopwatch_now - self.stopwatch_start
            imgui.text(f"time passed: " + str(delta_time))


    def on_update(self):
        if self.stopwatch:
            self.stopwatch_now = datetime.datetime.now()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()  # ask the dataset_in attribute for the node that it has got incoming; i.e., the dataset output attribute it is connected to. If it is not connected, a Node of type NullNode is returned - which eveluates to 'False'
        if data_source:
            input_frame = data_source.get_image(idx)  # get the incoming image (a Frame object), and call its .load() method to get the image's pixel data as a numpy array. The .load() method does some internal optimizations to ensure the image is only ever loaded from disk once, but the raw data remains intact as well.
            input_pxd = input_frame.load()
            out_frame = input_frame.clone()
            out_frame.data = np.random.normal(0, self.noise_stdev, input_pxd.shape)
            return out_frame
        else:
            out_frame = Frame(path = "virtual frame")  # create a new Frame object. Since it is not part of a raw data source, give it a fake path 'virtual frame'
            out_frame.data = np.random.normal(0, self.noise_stdev, (512, 512))  # no input image: output a 512 x 512 pixel noise image.
            return out_frame

