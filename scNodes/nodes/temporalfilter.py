from scNodes.core.node import *


def create():
    return TemporalFilterNode()


class TemporalFilterNode(Node):
    title = "Temporal filter"
    group = "Image processing"
    colour = (55 / 255, 236 / 255, 54 / 255, 1.0)
    sortid = 102

    FILTERS = ["Forward difference", "Backward difference", "Central difference", "Grouped difference", "Windowed average"]
    NEGATIVE_MODES = ["Absolute", "Zero", "Retain"]
    INCOMPLETE_GROUP_MODES = ["Discard"]

    def __init__(self):
        super().__init__()  # Was: super(LoadDataNode, self).__init__()
        self.size = 250

        # Set up connectable attributes
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)

        self.params["filter"] = 0
        self.params["negative_handling"] = 1
        self.params["incomplete_group_handling"] = 0
        self.params["skip"] = 1
        self.params["group_size"] = 11
        self.params["group_background_index"] = 1
        self.params["window"] = 3

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_end()
            self.connectable_attributes["dataset_in"].render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(170)
            _c, self.params["filter"] = imgui.combo("Filter", self.params["filter"], TemporalFilterNode.FILTERS)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(110)
            _c, self.params["negative_handling"] = imgui.combo("Negative handling", self.params["negative_handling"], TemporalFilterNode.NEGATIVE_MODES)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(90)
            if self.params["filter"] == 0 or self.params["filter"] == 1 or self.params["filter"] == 2:
                _c, self.params["skip"] = imgui.input_int("Step (# frames)", self.params["skip"], 0, 0)
                self.any_change = self.any_change or _c
            elif self.params["filter"] == 3:
                _c, self.params["group_size"] = imgui.input_int("Images per cycle", self.params["group_size"], 0, 0)
                self.any_change = self.any_change or _c
                _c, self.params["group_background_index"] = imgui.input_int("Background index", self.params["group_background_index"], 0, 0)
                self.any_change = self.any_change or _c
                _c, self.params["incomplete_group_handling"] = imgui.combo("Incomplete groups", self.params["incomplete_group_handling"], TemporalFilterNode.INCOMPLETE_GROUP_MODES)
                self.any_change = self.any_change or _c
            elif self.params["filter"] == 4:
                _c, self.params["window"] = imgui.input_int("Window size", self.params["window"], 0, 0)
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.connectable_attributes["dataset_in"].get_incoming_node()
        if data_source:
            pxd = None
            if self.params["filter"] == 0:
                pxd = data_source.get_image(idx + self.params["skip"]).load() - data_source.get_image(idx).load()
            elif self.params["filter"] == 1:
                pxd = data_source.get_image(idx).load() - data_source.get_image(idx - self.params["skip"]).load()
            elif self.params["filter"] == 2:
                pxd = data_source.get_image(idx + self.params["skip"]).load() - data_source.get_image(idx - self.params["skip"]).load()
            elif self.params["filter"] == 3:
                pxd = data_source.get_image(idx // self.params["group_size"]).load() - data_source.get_image(self.params["group_size"] * (idx // self.params["group_size"]) + self.params["group_background_index"]).load()
            elif self.params["filter"] == 4:
                pxd = np.zeros_like(data_source.get_image(idx).load())
                for i in range(-self.params["window"], self.params["window"] + 1):
                    pxd += data_source.get_image(idx + i).load()
                pxd /= (2 * self.params["window"] + 1)

            if self.params["negative_handling"] == 0:
                pxd = np.abs(pxd)
            elif self.params["negative_handling"] == 1:
                pxd[pxd < 0] = 0
            elif self.params["negative_handling"] == 2:
                pass

            out_image = data_source.get_image(idx)
            out_image.data = pxd
            return out_image
