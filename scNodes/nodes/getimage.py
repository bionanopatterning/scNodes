from scNodes.core.node import *


def create():
    return GetImageNode()


class GetImageNode(Node):
    title = "Dataset to image"
    group = "Converters"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    sortid = 2001
    IMAGE_MODES = ["By frame", "Time projection"]
    PROJECTIONS = ["Average", "Minimum", "Maximum", "St. dev."]

    def __init__(self):
        super().__init__()
        self.size = 200
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT,parent=self)
        self.connectable_attributes["image_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent=self)

        self.params["mode"] = 0
        self.params["projection"] = 0
        self.params["frame"] = 0
        self.image = None
        self.load_data_source = None
        self.params["pixel_size"] = 1

        self.roi = [0, 0, 0, 0]

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["image_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["image_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(140)
            _c, self.params["mode"]= imgui.combo("Mode", self.mode, GetImageNode.IMAGE_MODES)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(80)
            if self.params["mode"]== 0:
                _c, self.params["frame"] = imgui.input_int("Frame nr.", self.params["frame"], 0, 0)
                self.any_change = self.any_change or _c
            elif self.params["mode"]== 1:
                _c, self.params["projection"] = imgui.combo("Projection", self.params["projection"], GetImageNode.PROJECTIONS)
                if _c:
                    self.image = None
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            if self.any_change:
                self.configure_settings()

            super().render_end()

    def configure_settings(self):
        datasource = self.connectable_attributes["dataset_in"].get_incoming_node()
        if datasource:
            try:
                if cfg.profiling:
                    self.profiler_count += 1
                if self.params["mode"]== 0:
                    image_in = datasource.get_image(self.params["frame"])
                    self.image = image_in.load()
                    self.params["pixel_size"] = image_in.pixel_size
                elif self.params["mode"]== 1:
                    load_data_node = Node.get_source_load_data_node(self)
                    self.params["pixel_size"] = load_data_node.dataset.pixel_size
                    load_data_node.load_on_the_fly = False
                    self.load_data_source = load_data_node
            except Exception as e:
                cfg.set_error(e, "GetImageNode error upon attempting to gen img.\n"+str(e))
        else:
            cfg.set_error(Exception(), "GetImageNode missing input dataset.")
        self.any_change = True

    def on_update(self):
        if self.params["mode"]== 1 and self.image is None:
            if self.load_data_source is not None:
                if self.load_data_source.done_loading:
                    self.generate_projection()

    def generate_projection(self):
        data_source = self.connectable_attributes["dataset_in"].get_incoming_node()
        frame = data_source.get_image(0)
        n_frames = Node.get_source_load_data_node(self).dataset.n_frames
        projection_image = np.zeros((frame.width, frame.height, n_frames))
        for i in range(n_frames):
            projection_image[:, :, i] = data_source.get_image(i).load()
        if self.params["projection"] == 0:
            self.image = np.average(projection_image, axis = 2)
        elif self.params["projection"] == 1:
            self.image = np.min(projection_image, axis = 2)
        elif self.params["projection"] == 2:
            self.image = np.max(projection_image, axis = 2)
        elif self.params["projection"] == 3:
            self.image = np.std(projection_image, axis = 2)
        self.any_change = True

    def get_image_impl(self, idx=None):
        if cfg.profiling:
            self.profiler_count -= 1
        if self.any_change:
            self.configure_settings()
        if self.image is not None:
            out_frame = Frame("virtual_frame")
            out_frame.data = self.image
            out_frame.pixel_size = self.params["pixel_size"]
            return out_frame
