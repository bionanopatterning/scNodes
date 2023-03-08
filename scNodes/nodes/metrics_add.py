from scNodes.core.node import *
from scipy import stats

def create():
    return AddMetric()


class AddMetric(Node):
    title = "Add metric"
    colour = (145/255, 236/255, 54/255, 1.0)
    group = "Metrics"
    size = 190
    sortid = 951
    #enabled = False

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


