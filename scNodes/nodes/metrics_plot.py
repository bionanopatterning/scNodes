import imgui

from scNodes.core.node import *

def create():
    return PlotMetricsNode()

class PlotMetricsNode(Node):
    title = "Plot metric"
    colour = (145 / 255, 236 / 255, 54 / 255, 1.0)
    group = "Metrics"
    size = 190
    sortid = 952

    def __init__(self):
        super().__init__()

        self.buffer_last_output = True

        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)

        self.params["selected_metric"] = ""
        self.params["range_min"] = 0
        self.params["range_max"] = 1
        self.params["to_plot"] = dict()

        self.plot_list = list()
        self.metric_list = list()

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_in"].render_end()

            self.metric_list = []
            if self.last_frame_returned is not None:
                self.metric_list = list(self.last_frame_returned.scalar_metrics.keys())

            if len(self.plot_list) != len(self.metric_list):
                self.plot_list = [False] * len(self.metric_list)

            imgui.text("Metrics to plot:")
            for i in range(len(self.metric_list)):
                checkbox_label = self.metric_list[i] if len(self.metric_list) < 21 else self.metric_list[i][:19]+"..."
                _c, self.plot_list[i] = imgui.checkbox(checkbox_label, self.plot_list[i])
            if len(self.metric_list) == 0:
                imgui.text(" <no metrics available>")
            imgui.spacing()
            imgui.spacing()

            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2.0, 2.0))
            imgui.text("Frames to plot:")
            imgui.new_line()
            imgui.set_next_item_width(60)
            _c, self.params["range_min"] = imgui.input_int("##min", self.params["range_min"], 0.0, 0.0)
            imgui.same_line()
            imgui.text(" to ")
            imgui.same_line()
            imgui.set_next_item_width(60)
            _c, self.params["range_max"] = imgui.input_int("##max", self.params["range_max"], 0.0, 0.0)
            self.tooltip("The max frame is included, e.g. '0 to 10' gives 11 frames.")
            imgui.pop_style_var(1)
            _cw = imgui.get_content_region_available_width()
            imgui.new_line()
            imgui.same_line(spacing = (_cw - 50) / 2)
            if imgui.button("Plot", width = 50, height = 20):
                self.plot()

            super().render_end()

    def plot(self):
        source = self.connectable_attributes["dataset_in"].get_incoming_node()
        frames = np.arange(self.params["range_min"], self.params["range_max"]+1)
        vals = dict()
        for i in range(len(self.metric_list)):
            if self.plot_list[i]:
                vals[self.metric_list[i]] = []
        for f in frames:
            frame = source.get_image(f)
            for i in range(len(self.metric_list)):
                metric_name = self.metric_list[i]
                if self.plot_list[i]:
                    vals[metric_name].append(frame.scalar_metrics[metric_name])

        for metric in vals:
            plt.plot(frames, vals[metric], linewidth = 1, label=metric)
        plt.xlim([frames[0], frames[-1]])
        plt.legend()
        plt.show()

    def get_image_impl(self, idx=None):
        # just forward the frame here. Since 'buffer_last_output' is True, we can
        # access 'self.last_frame_returned.scalar_metrics' to get the dict of
        # all metrics associated with the frame.
        source = self.connectable_attributes["dataset_in"].get_incoming_node()
        if source:
            return source.get_image(idx)
