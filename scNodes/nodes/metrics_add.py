from scNodes.core.node import *
from scipy import stats

def create():
    return AddMetric()


class AddMetric(Node):
    description = "Add a 'metric' to a dataset. Metrics are named tags that hold a certain value - for example, the\n" \
                  "Registration node adds a metric called 'Drift (nm)' to every frame that it processes.\n" \
                  "\n" \
                  "Using the Add Metric node, you can add extra metrics to frames at any point in the processing pi-\n" \
                  "peline. For example, we can calculate the mean of a frame's pixel intensities after a background\n" \
                  "subtraction processing step, and call this metric 'filter_1' If the mean is not close to zero,\n" \
                  "something might have gone wrong in the filtering step. This metric can then be used to either\n" \
                  "filter out frames that are flawed (using a 'Discard by metric' node), or it can be plotted to\n" \
                  "inspect the quality of the processing pipeline (using a 'Plot metric' node)."
    title = "Add metric"
    colour = (145/255, 236/255, 54/255, 1.0)
    group = "Metrics"
    size = 190
    sortid = 3951

    DEFAULT_METRICS = ["mean", "std", "1st moment", "2nd moment"]

    def __init__(self):
        super().__init__()

        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes["sample_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)

        self.params["metric_name"] = "Unnamed metric"
        self.params["metric_type"] = "mean"

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["dataset_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("Dataset to sample:")
            imgui.spacing()
            self.connectable_attributes["sample_in"].render_start()
            self.connectable_attributes["sample_in"].render_end()
            imgui.spacing()
            imgui.text("Metric name:")
            imgui.set_next_item_width(imgui.get_content_region_available_width())
            _c, self.params["metric_name"] = imgui.input_text("##name", self.params["metric_name"], 256)
            current_metric_index = AddMetric.DEFAULT_METRICS.index(self.params["metric_type"])
            imgui.set_next_item_width(120)
            _c, self.params["metric_type"] = imgui.combo("Metric", current_metric_index, AddMetric.DEFAULT_METRICS)
            self.params["metric_type"] = AddMetric.DEFAULT_METRICS[self.params["metric_type"]]
            super().render_end()

    def get_image_impl(self, idx=None):
        source = self.connectable_attributes["dataset_in"].get_incoming_node()
        if source:
            frame = source.get_image(idx)
            pxd = frame.load()

            if self.params["metric_type"] == "mean":
                frame_metric_value = np.mean(pxd)
            if self.params["metric_type"] == "std":
                frame_metric_value = np.std(pxd)
            if self.params["metric_type"] == "1st moment":
                frame_metric_value = stats.moment(pxd, moment=1, axis=None)
            if self.params["metric_type"] == "2nd moment":
                frame_metric_value = stats.moment(pxd, moment=2, axis=None)

            frame.scalar_metrics[self.params["metric_name"]] = frame_metric_value
            return frame


