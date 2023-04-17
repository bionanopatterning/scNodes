from scNodes.core.node import *


def create():
    return MathNode()


class MathNode(Node):
    description = "Perform various mathematical operations on the input images' pixel data, such as taking the logarithm."
    title = "Math"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    group = "Converters"
    sortid = 2000

    OPERATIONS = ["Power", "Log", "Invert", "Threshold", "Add constant", "Multiply", "Absolute"]

    def __init__(self):
        super().__init__()

        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.connectable_attributes["dataset_in"].colour = ConnectableAttribute.COLOUR[ConnectableAttribute.TYPE_DATASET]
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.OUTPUT, self, [ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.connectable_attributes["dataset_out"].colour = ConnectableAttribute.COLOUR[ConnectableAttribute.TYPE_DATASET]

        self.params["operation"] = 0

        # operation specific vars
        self.params["power"] = 2.0
        self.params["threshold"] = 100
        self.params["threshold_keep_high"] = True
        self.params["threshold_fill_value"] = 0
        self.params["threshold_make_mask"] = True
        self.params["constant"] = 100
        self.params["factor"] = 2.0

        self.params["last_image_min"] = 0
        self.params["last_image_max"] = 256

        self.params["replace_nan_by"] = 0
        self.params["replace_inf_by"] = 0
        self.params["replace_inf"] = False

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["dataset_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("Operation")
            imgui.set_next_item_width(170)
            _c, self.params["operation"] = imgui.combo("##operation", self.params["operation"], MathNode.OPERATIONS)
            self.any_change = self.any_change or _c
            imgui.push_item_width(120)
            if self.params["operation"] == 0:
                _c, self.params["power"] = imgui.slider_float("power", self.params["power"], -2.0, 2.0, format = "%.1f")
                self.any_change = self.any_change or _c
            elif self.params["operation"] == 3:
                _c, self.params["threshold"] = imgui.slider_float("threshold", self.params["threshold"], self.params["last_image_min"], self.params["last_image_max"], format = "%.0f")
                self.any_change = self.any_change or _c
                _c, self.params["threshold_make_mask"] = imgui.checkbox("output mask", self.params["threshold_make_mask"])
                self.any_change = self.any_change or _c
                if not self.params["threshold_make_mask"]:
                    _c, self.params["threshold_fill_value"] = imgui.slider_float("fill", self.params["threshold_fill_value"], self.params["last_image_min"], self.params["last_image_max"], format = "%.0f")
                    self.any_change = self.any_change or _c
                _c, self.params["threshold_keep_high"] = imgui.checkbox("keep low", self.params["threshold_keep_high"])
                Node.tooltip("By default, values lower than the threshold value are replaced/masked.\n"
                             "Check this box to invert the behaviour: values higher than the treshold\n"
                             "are replaced/masked instead.")
                self.any_change = self.any_change or _c
            elif self.params["operation"] == 4:
                _c, self.params["constant"] = imgui.slider_float("value", self.params["constant"], -1000, 1000, format = "%.0f")
                self.any_change = self.any_change or _c
            elif self.params["operation"] == 5:
                _c, self.params["factor"] = imgui.slider_float("factor", self.params["factor"], 0.0, 2.0, format = "%.2f")
                self.any_change = self.any_change or _c
            Node.tooltip("Use ctrl + click on a slider to override it and manually enter a value.")

            imgui.spacing()
            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()
            super().render_end()

    def render_advanced(self):
        imgui.push_item_width(40)
        imgui.text("Replace NaN by:")
        _c, self.params["replace_nan_by"] = imgui.input_float("##nan", self.params["replace_nan_by"], format = "%.0f")
        self.any_change = self.any_change or _c
        imgui.text("Replace inf by:")
        _c, self.params["replace_inf_by"] = imgui.input_float("##inf", self.params["replace_inf_by"], format = "%.0f")
        self.any_change = self.any_change or _c
        imgui.same_line()
        _c, self.params["replace_inf"] = imgui.checkbox("replace inf", self.params["replace_inf"])
        self.any_change = self.any_change or _c
        imgui.pop_item_width()

    def get_image_impl(self, idx):
        datasource = self.connectable_attributes["dataset_in"].get_incoming_node()
        if datasource:
            img_in = datasource.get_image(idx)
            pxd = img_in.load()

            if self.params["operation"] == 0:
                pxd = pxd ** self.params["power"]
            elif self.params["operation"] == 1:
                pxd = np.log(pxd)
            elif self.params["operation"] == 2:
                pxd = - pxd
            elif self.params["operation"] == 3:
                if self.FRAME_REQUESTED_BY_IMAGE_VIEWER:
                    self.params["last_image_min"] = np.amin(pxd)
                    self.params["last_image_max"] = np.amax(pxd)
                if self.params["threshold_keep_high"]:
                    if self.params["threshold_make_mask"]:
                        pxd = pxd < self.params["threshold"]
                    else:
                        pxd[pxd < self.params["threshold"]] = self.params["threshold_fill_value"]
                else:
                    if self.params["threshold_make_mask"]:
                        pxd = pxd >= self.params["threshold"]
                    else:
                        pxd[pxd >= self.params["threshold"]] = self.params["threshold_fill_value"]
            elif self.params["operation"] == 4:
                pxd += self.params["constant"]
            elif self.params["operation"] == 5:
                pxd *= self.params["factor"]
            elif self.params["operation"] == 6:
                pxd = np.abs(pxd)

            # replace nan and inf
            _inf_val = self.params["replace_inf_by"] if self.params["replace_inf"] else None
            np.nan_to_num(pxd, copy=False, nan=self.params["replace_nan_by"], posinf = _inf_val, neginf = _inf_val)
            img_out = img_in.clone()
            img_out.data = pxd
            return img_out