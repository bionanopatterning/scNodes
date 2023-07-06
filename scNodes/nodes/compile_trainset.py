from scNodes.core.node import *
import scNodes.core.widgets as widgets
from tkinter import filedialog


def create():
    return CompileTrainsetNode()


class CompileTrainsetNode(Node):
    description = "TODO"
    title = "Compile dataset"
    group = "Neural networks"
    colour = (138 / 255, 200 / 255, 186 / 255, 1.0)

    def __init__(self):
        super().__init__()
        self.size = 180
        self.connectable_attributes["input_a"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["input_b"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.params["box_size"] = 64
        self.params["n_samples"] = 128
        self.processes = list()

    def render(self):
        if super().render_start():
            imgui.text("Model input:")
            imgui.new_line()
            self.connectable_attributes["input_a"].render_start()
            self.connectable_attributes["input_a"].render_end()

            imgui.text("Model output:")
            imgui.new_line()
            self.connectable_attributes["input_b"].render_start()
            self.connectable_attributes["input_b"].render_end()

            imgui.push_item_width(35)
            _, self.params["box_size"] = imgui.input_int("Box size", self.params["box_size"], 0, 0)
            _, self.params["n_samples"] = imgui.input_int("N samples", self.params["n_samples"], 0, 0)
            imgui.pop_item_width()

            if widgets.centred_button("Start", 60, 20, 10):
                path = filedialog.asksaveasfilename(filetypes=[("scNodes traindata", cfg.filetype_traindata)])
                self.processes.append(BackgroundProcess(self.compile, (path+cfg.filetype_traindata, self.params["n_samples"], self.params["box_size"])))
                self.processes[-1].start()

            for p in self.processes:
                self.progress_bar(p.progress)
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
                if p.progress >= 1.0:
                    self.processes.remove(p)
            super().render_end()

    def compile(self, path, n_samples, box_size, process):
        # check input image sizes
        src_x = self.connectable_attributes["input_a"].get_incoming_node()
        src_y = self.connectable_attributes["input_b"].get_incoming_node()
        sx = src_x.get_image(0).load().shape
        sy = src_y.get_image(0).load().shape
        if sx != sy:
            process.set_progress(1.0)
            return

        n_max = Node.get_source_load_data_node(self).dataset.n_frames

        f = np.sort(np.random.randint(0, n_max, n_samples))
        x = np.random.randint(0, sx[0] - box_size, n_samples)
        y = np.random.randint(0, sy[1] - box_size, n_samples)

        current_frame = -1
        img_x = None
        img_y = None
        out_x = list()
        out_y = list()
        for i in range(n_samples):
            if f[i] != current_frame:
                current_frame = f[i]
                img_x = src_x.get_image(current_frame).load()
                img_y = src_y.get_image(current_frame).load()
            out_x.append(img_x[x[i]:x[i]+box_size, y[i]:y[i]+box_size])
            out_y.append(img_y[x[i]:x[i] + box_size, y[i]:y[i] + box_size])
            process.set_progress(i / (n_samples - 1))
        all_out = np.array(out_x + out_y)
        tifffile.imwrite(path, all_out)






