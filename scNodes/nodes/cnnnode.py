from scNodes.core.node import *
import scNodes.core.widgets as widgets
import os
import importlib
from tensorflow.keras.callbacks import Callback


def create():
    return CNNNode()


class CNNNode(Node):
    description = "TODO"
    title = "Neural network"
    group = "Neural networks"
    colour = (138 / 255, 200 / 255, 186 / 255, 1.0)

    MODEL_DICT = dict()
    MODEL_NAMES = list()

    def __init__(self):
        super().__init__()
        self.size = 210

        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)

        self.params["model_enum"] = 0
        self.params["train_data_path"] = "..."
        self.params["epochs"] = 25
        self.params["batch_size"] = 32
        self.params["overlap"] = 0.5
        self.params["loss"] = -1.0
        self.params["model_name"] = ""
        self.model = None
        self.compiled = False
        self.process = None
        self.box_size = -1

        if len(CNNNode.MODEL_NAMES) == 0:
            CNNNode.load_models()

    @staticmethod
    def load_models():
        model_files = glob.glob(os.path.join(cfg.root, "nodes", "cnns", "*.py"))
        for file in model_files:
            try:
                module_name = os.path.basename(file)[:-3]
                mod = importlib.import_module(("scNodes." if not cfg.frozen else "") + "nodes.cnns." + module_name)
                CNNNode.MODEL_DICT[mod.title] = mod.create
            except Exception as e:
                cfg.set_error(e, "Could not load CNN at path: " + file)
        CNNNode.MODEL_NAMES = list(CNNNode.MODEL_DICT.keys())

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            if self.compiled:
                self.connectable_attributes["dataset_out"].render_start()
                self.connectable_attributes["dataset_out"].render_end()


            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if imgui.begin_tab_bar("##tabs"):
                if imgui.begin_tab_item("Training")[0]:
                    imgui.push_item_width(imgui.get_content_region_available_width())
                    if not self.compiled:
                        imgui.text("Select model:")
                        _, self.params["model_enum"] = imgui.combo("##model", self.params["model_enum"], CNNNode.MODEL_NAMES)
                    else:
                        imgui.text(self.params["model_name"] + f"\nloss: {self.params['loss']:.3f}")
                    _, self.params["train_data_path"] = widgets.select_file("...", self.params["train_data_path"], filetypes=[("scNodes traindata",cfg.filetype_traindata)])
                    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2, 2))
                    imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 10)
                    _, self.params["epochs"] = imgui.slider_int("##epochs", self.params["epochs"], 1, 64, f'{self.params["epochs"]} epochs')
                    _, self.params["batch_size"] = imgui.slider_int("##batchsize", self.params["batch_size"], 1, 64, f'{self.params["batch_size"]} batch size')

                    imgui.pop_item_width()

                    cw = imgui.get_content_region_available_width()
                    if imgui.button("load", (cw - 16) / 3, 20):
                        pass
                    imgui.same_line()
                    if imgui.button("save", (cw - 16) / 3, 20):
                        pass
                    imgui.same_line()
                    if imgui.button("train", (cw - 16) / 3, 20):
                        self.process = BackgroundProcess(self.train, ())
                        self.process.start()

                    imgui.pop_style_var(3)
                    imgui.end_tab_item()
                if imgui.begin_tab_item("Application")[0]:
                    if self.compiled:
                        imgui.text(self.params["model_name"] + f"\nloss: {self.params['loss']:.3f}")
                    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2, 2))
                    imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 10)
                    imgui.push_item_width(imgui.get_content_region_available_width())
                    _c, self.params["overlap"] = imgui.slider_float("##overlap", self.params["overlap"], 0.0, 0.9, f'{self.params["overlap"]:.2f} overlap')
                    self.mark_change(_c)
                    imgui.pop_item_width()
                    imgui.pop_style_var(3)
                    imgui.end_tab_item()
                imgui.end_tab_bar()

            if self.process is not None:
                self.progress_bar(self.process.progress)
                if self.process.progress >= 1.0:
                    self.process = None
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()

            super().render_end()

    def compile(self, box_size):
        model_module_name = CNNNode.MODEL_NAMES[self.params["model_enum"]]
        self.model = CNNNode.MODEL_DICT[model_module_name]((box_size, box_size, 1))
        self.params["model_name"] = CNNNode.MODEL_NAMES[self.params["model_enum"]]
        self.compiled = True
        self.box_size = box_size

    def train(self, process):
        try:
            train_data = tifffile.imread(self.params["train_data_path"])
            L = train_data.shape[0]
            train_x = train_data[:L//2, :, :, None]
            train_y = train_data[L//2:, :, :, None]
            n_samples = train_x.shape[0]
            box_size = train_x.shape[1]
            # compile, if not compiled yet
            if not self.compiled:
                self.compile(box_size)
            # if training data box size is not compatible with the compiled model, abort.
            if box_size != self.box_size:
                return

            # train
            self.model.fit(train_x, train_y, epochs=self.params["epochs"], batch_size=self.params["batch_size"], shuffle=True,
                           callbacks=[TrainingProgressCallback(self.process, n_samples, self.params["batch_size"], self)])
        except Exception as e:
            cfg.set_error(e, "Could not train model - see details below.")
            self.process.stop()

    def apply_to_image(self, img):
        def tile(img):
            w, h = img.shape
            pad_w = self.box_size - (w % self.box_size)
            pad_h = self.box_size - (h % self.box_size)
            stride = int(self.box_size * (1.0 - self.params["overlap"]))
            tiles = list()
            img = np.pad(img, ((0, pad_w), (0, pad_h)))
            for x in range(0, w + pad_w - self.box_size + 1, stride):
                for y in range(0, h + pad_h - self.box_size + 1, stride):
                    tiles.append(img[x:x+self.box_size, y:y+self.box_size])

            return np.array(tiles), (w, h), (pad_w, pad_h), stride

        def detile(tiles, image_size, padding, stride):
            pad_w, pad_h = padding
            w, h = image_size
            out_image = np.zeros((w + pad_w, h + pad_h))
            count = np.zeros((w + pad_w, h + pad_h), dtype=int)
            i = 0
            for x in range(0, w + pad_w - self.box_size + 1, stride):
                for y in range(0, h + pad_h - self.box_size + 1, stride):
                    out_image[x:x + self.box_size, y:y + self.box_size] += tiles[i]
                    count[x:x + self.box_size, y:y + self.box_size] += 1
                    i += 1
            count[count == 0] = 1
            out_image /= count
            out_image = out_image[:w, :h]
            return out_image

        boxes, image_size, padding, stride = tile(img)
        out_boxes = np.squeeze(self.model.predict(boxes))
        out_img = detile(out_boxes, image_size, padding, stride)
        return out_img

    def get_image_impl(self, idx):
        if not self.compiled:
            return None
        src = self.connectable_attributes["dataset_in"].get_incoming_node()
        if src:
            frame_in = src.get_image(idx)
            pxd = frame_in.load()
            frame_out = frame_in.clone()
            frame_out.data = self.apply_to_image(pxd)
            return frame_out
        # apply CNN to image

class TrainingProgressCallback(Callback):
    def __init__(self, process, n_samples, batch_size, node):
        super().__init__()
        self.process = process
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.batches_in_epoch = 0
        self.current_epoch = 0
        self.node = node

    def on_epoch_begin(self, epoch, logs=None):
        self.batches_in_epoch = self.n_samples // self.batch_size
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        progress_in_current_epoch = (batch + 1) / self.batches_in_epoch
        total_progress = (self.current_epoch + progress_in_current_epoch) / self.params['epochs']
        self.process.set_progress(total_progress)
        self.node.params["loss"] = logs["loss"]
