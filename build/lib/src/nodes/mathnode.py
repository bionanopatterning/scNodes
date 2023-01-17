from scNodes.core.node import *


def create():
    return MathNode()


class MathNode(Node):
    title = "Math"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    group = "Converters"
    sortid = 2000

    OPERATIONS = ["Power", "Log", "Invert", "Threshold", "Add constant", "Multiply"]

    def __init__(self):
        super().__init__()

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.dataset_in.colour = ConnectableAttribute.COLOUR[ConnectableAttribute.TYPE_DATASET]
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.OUTPUT, self, [ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.dataset_out.colour = ConnectableAttribute.COLOUR[ConnectableAttribute.TYPE_DATASET]

        self.operation = 0

        # operation specific vars
        self.power = 2.0
        self.threshold = 100
        self.threshold_keep_high = True
        self.threshold_fill_value = 0
        self.constant = 100
        self.factor = 2.0

        self.last_image_min = 0
        self.last_image_max = 256

        self.replace_nan_by = 0
        self.replace_inf_by = 0
        self.replace_inf = False

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_out.render_start()
            self.dataset_in.render_end()
            self.dataset_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("Operation")
            imgui.set_next_item_width(170)
            _c, self.operation = imgui.combo("##operation", self.operation, MathNode.OPERATIONS)
            self.any_change = self.any_change or _c
            imgui.push_item_width(120)
            if self.operation == 0:
                _c, self.power = imgui.slider_float("power", self.power, -2.0, 2.0, format = "%.1f")
                self.any_change = self.any_change or _c
            elif self.operation == 3:
                _c, self.threshold = imgui.slider_float("threshold", self.threshold, self.last_image_min, self.last_image_max, format = "%.0f")
                self.any_change = self.any_change or _c
                _c, self.threshold_fill_value = imgui.slider_float("fill", self.threshold_fill_value, self.last_image_min, self.last_image_max, format = "%.0f")
                self.any_change = self.any_change or _c
                _c, self.threshold_keep_high = imgui.checkbox("keep low", self.threshold_keep_high)
                Node.tooltip("By default, values lower than the threshold value are replaced.\n"
                             "Check this box to invert the behaviour: values higher than the\n"
                             "threshold are replaced instead.")
                self.any_change = self.any_change or _c
            elif self.operation == 4:
                _c, self.constant = imgui.slider_float("value", self.constant, -1000, 1000, format = "%.0f")
                self.any_change = self.any_change or _c
            elif self.operation == 5:
                _c, self.factor = imgui.slider_float("factor", self.factor, 0.0, 2.0, format = "%.2f")
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
        _c, self.replace_nan_by = imgui.input_float("##nan", self.replace_nan_by, format = "%.0f")
        self.any_change = self.any_change or _c
        imgui.text("Replace inf by:")
        _c, self.replace_inf_by = imgui.input_float("##inf", self.replace_inf_by, format = "%.0f")
        self.any_change = self.any_change or _c
        imgui.same_line()
        _c, self.replace_inf = imgui.checkbox("replace inf", self.replace_inf)
        self.any_change = self.any_change or _c
        imgui.pop_item_width()

    def get_image_impl(self, idx):
        datasource = self.dataset_in.get_incoming_node()
        if datasource:
            img_in = datasource.get_image(idx)
            pxd = img_in.load()

            if self.operation == 0:
                pxd = pxd ** self.power
            elif self.operation == 1:
                pxd = np.log(pxd)
            elif self.operation == 2:
                pxd = - pxd
            elif self.operation == 3:
                if self.FRAME_REQUESTED_BY_IMAGE_VIEWER:
                    self.last_image_min = np.amin(pxd)
                    self.last_image_max = np.amax(pxd)
                if self.threshold_keep_high:
                    pxd[pxd < self.threshold] = self.threshold_fill_value
                else:
                    pxd[pxd >= self.threshold] = self.threshold_fill_value
            elif self.operation == 4:
                pxd += self.constant
            elif self.operation == 5:
                pxd *= self.factor

            # replace nan and inf
            _inf_val = self.replace_inf_by if self.replace_inf else None
            np.nan_to_num(pxd, copy=False, nan=self.replace_nan_by, posinf = _inf_val, neginf = _inf_val)
            img_out = img_in.clone()
            img_out.data = pxd
            return img_out