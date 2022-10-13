from node import *


def create():
    return TemporalFilterNode()


class TemporalFilterNode(Node):
    title = "Temporal filter"
    group = "Image processing"
    colour = (55 / 255, 236 / 255, 54 / 255, 1.0)

    FILTERS = ["Forward difference", "Backward difference", "Central difference", "Grouped difference", "Windowed average"]
    NEGATIVE_MODES = ["Absolute", "Zero", "Retain"]
    INCOMPLETE_GROUP_MODES = ["Discard"]

    def __init__(self):
        super().__init__(Node.TYPE_TEMPORAL_FILTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [250, 220]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

        self.filter = 0
        self.negative_handling = 1
        self.incomplete_group_handling = 0
        self.skip = 1
        self.group_size = 11
        self.group_background_index = 1
        self.window = 3

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(170)
            _c, self.filter = imgui.combo("Filter", self.filter, TemporalFilterNode.FILTERS)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(110)
            _c, self.negative_handling = imgui.combo("Negative handling", self.negative_handling, TemporalFilterNode.NEGATIVE_MODES)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(90)
            if self.filter == 0 or self.filter == 1 or self.filter == 2:
                _c, self.skip = imgui.input_int("Step (# frames)", self.skip, 0, 0)
                self.any_change = self.any_change or _c
            elif self.filter == 3:
                _c, self.group_size = imgui.input_int("Images per cycle", self.group_size, 0, 0)
                self.any_change = self.any_change or _c
                _c, self.group_background_index = imgui.input_int("Background index", self.group_background_index, 0, 0)
                self.any_change = self.any_change or _c
                _c, self.incomplete_group_handling = imgui.combo("Incomplete groups", self.incomplete_group_handling, TemporalFilterNode.INCOMPLETE_GROUP_MODES)
                self.any_change = self.any_change or _c
            elif self.filter == 4:
                _c, self.window = imgui.input_int("Window size", self.window, 0, 0)
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            pxd = None
            if self.filter == 0:
                pxd = data_source.get_image(idx + self.skip).load() - data_source.get_image(idx).load()
            elif self.filter == 1:
                pxd = data_source.get_image(idx).load() - data_source.get_image(idx - self.skip).load()
            elif self.filter == 2:
                pxd = data_source.get_image(idx + self.skip).load() - data_source.get_image(idx - self.skip).load()
            elif self.filter == 3:
                pxd = data_source.get_image(idx // self.group_size).load() - data_source.get_image(self.group_size * (idx // self.group_size) + self.group_background_index).load()
            elif self.filter == 4:
                pxd = np.zeros_like(data_source.get_image(idx).load())
                for i in range(-self.window, self.window + 1):
                    pxd += data_source.get_image(idx + i).load()
                pxd /= (2 * self.window + 1)

            if self.negative_handling == 0:
                pxd = np.abs(pxd)
            elif self.negative_handling == 1:
                pxd[pxd < 0] = 0
            elif self.negative_handling == 2:
                pass

            out_image = data_source.get_image(idx)
            out_image.data = pxd
            return out_image
