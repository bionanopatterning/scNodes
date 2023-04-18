from scNodes.core.node import *


def create():
    return DiscardFramesNode()


class DiscardFramesNode(Node):
    description = "Discard frames based on the value of any metric associated with the frame. Discarded frames are\n" \
                  "ignored in the PSF-fitting node - they do not contribute to the final reconstruction.\n" \
                  "\n" \
                  "See the 'Add metric' node description for more information on metrics."
    title = "Discard by metric"
    colour = (145/255, 236/255, 54/255, 1.0)
    group = "Metrics"
    size = 190
    sortid = 3953

    DEFAULT_METRICS = ["mean", "std"]

    def __init__(self):
        super().__init__()

        self.buffer_last_output = True

        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)

        self.params["selected_metric"] = "Select metric"
        self.params["range_min"] = -1
        self.params["range_max"] = 1

        self.current_frame_metric_value = 0

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["dataset_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # get a list of available metrics
            metric_list = [] + DiscardFramesNode.DEFAULT_METRICS

            if self.last_frame_returned is not None:
                metric_list += list(self.last_frame_returned.scalar_metrics.keys())

            if self.params["selected_metric"] not in metric_list:
                self.params["selected_metric"] = metric_list[0]
            current_idx = metric_list.index(self.params["selected_metric"])

            _cw = imgui.get_content_region_available_width()
            imgui.set_next_item_width(_cw - 70)
            _c, self.params["selected_metric"] = imgui.combo("Metric", current_idx, metric_list)
            self.mark_change(_c)
            self.params["selected_metric"] = metric_list[self.params["selected_metric"]]

            imgui.spacing()

            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2.0, 2.0))
            imgui.text("Minimum value:")
            imgui.same_line()
            imgui.set_next_item_width(65)
            _c, self.params["range_min"] = imgui.drag_float("##min_float", self.params["range_min"], 1.0, 0.0, 0.0, '%.2f')
            self.mark_change(_c and not imgui.is_item_active())  # only fire when value is done being edited
            self.tooltip("double click to enter a value, or drag to adjust.")

            imgui.text("Maximum value:")
            imgui.same_line()
            imgui.set_next_item_width(65)
            _c, self.params["range_max"] = imgui.drag_float("##max_float", self.params["range_max"], 1.0, 0.0, 0.0, '%.2f')
            if imgui.is_item_active():
                _c = False  # only update frame when drag float is released
            self.tooltip("double click to enter a value, or drag to adjust.")
            self.mark_change(_c and not imgui.is_item_active())  # only fire when value is done being edited
            imgui.pop_style_var(1)

            imgui.text("Current frame: ")
            imgui.same_line()
            pop_color = False
            if not (self.params["range_min"] < self.current_frame_metric_value < self.params["range_max"]):
                imgui.push_style_color(imgui.COLOR_TEXT, 0.8, 0.1, 0.1, 1.0)
                pop_color = True
            imgui.text(f"{self.current_frame_metric_value:.2f}")
            if pop_color:
                imgui.pop_style_color(1)

            super().render_end()

    def get_image_impl(self, idx=None):
            source = self.connectable_attributes["dataset_in"].get_incoming_node()
            if source:
                frame = source.get_image(idx)
                pxd = frame.load()

                if self.params["selected_metric"] == "mean":
                    frame_metric_value = np.mean(pxd)
                elif self.params["selected_metric"] == "std":
                    frame_metric_value = np.std(pxd)
                else:
                    frame_metric_value = frame.scalar_metrics[self.params["selected_metric"]]
                self.current_frame_metric_value = frame_metric_value
                frame.discard = not (self.params["range_min"] < frame_metric_value < self.params["range_max"])
                return frame


