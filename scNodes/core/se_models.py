from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import tifffile
import mrcfile
import numpy as np
from itertools import count
import glob
import os
import scNodes.core.config as cfg
import importlib
import threading
import json
from scNodes.core.opengl_classes import Texture


class SEModel:
    idgen = count(0)
    AVAILABLE_MODELS = []
    MODELS = dict()
    MODELS_LOADED = False
    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 266 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]

    def __init__(self):
        self.uid = next(SEModel.idgen)
        self.title = "Unnamed model"
        self.colour = SEModel.DEFAULT_COLOURS[self.uid % len(SEModel.DEFAULT_COLOURS)]
        self.apix = -1.0
        self.compiled = False
        self.box_size = -1
        self.model = None
        self.model_enum = 0
        self.epochs = 25
        self.batch_size = 32
        self.train_data_path = "..."
        self.active = True
        self.blend = False
        self.show = True
        self.alpha = 0.75
        self.threshold = 0.5
        self.overlap = 0.5
        self.active_tab = 0
        self.background_process = None
        self.n_parameters = 0

        self.data = None
        self.texture = Texture(format="r32f")
        self.texture.set_linear_mipmap_interpolation()
        # openGL stuff

        if not SEModel.MODELS_LOADED:
            SEModel.load_models()

    def delete(self):
        pass

    def save(self, file_path):
        # Split the file_path into directory and file
        directory = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        # Save the Keras model
        model_path = os.path.join(directory, base_name + '_weights.h5')
        self.model.save(model_path)

        # Save metadata
        metadata = {
            'uid': self.uid,
            'title': self.title,
            'colour': self.colour,
            'apix': self.apix,
            'compiled': self.compiled,
            'box_size': self.box_size,
            'model_enum': self.model_enum,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'train_data_path': self.train_data_path,
            'active': self.active,
            'blend': self.blend,
            'show': self.show,
            'alpha': self.alpha,
            'threshold': self.threshold,
            'overlap': self.overlap,
            'active_tab': self.active_tab,
            'n_parameters': self.n_parameters
        }
        with open(file_path, 'w') as f:
            json.dump(metadata, f)

    def load(self, file_path):
        # Split the file_path into directory and file
        directory = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)

        # Load the Keras model
        model_path = os.path.join(directory, base_name + '_weights.h5')
        self.model = tf.keras.models.load_model(model_path)

        # Load metadata
        with open(file_path, 'r') as f:
            metadata = json.load(f)
        self.uid = metadata['uid']
        self.title = metadata['title']
        self.colour = metadata['colour']
        self.apix = metadata['apix']
        self.compiled = metadata['compiled']
        self.box_size = metadata['box_size']
        self.model_enum = metadata['model_enum']
        self.epochs = metadata['epochs']
        self.batch_size = metadata['batch_size']
        self.train_data_path = metadata['train_data_path']
        self.active = metadata['active']
        self.blend = metadata['blend']
        self.show = metadata['show']
        self.alpha = metadata['alpha']
        self.threshold = metadata['threshold']
        self.overlap = metadata['overlap']
        self.active_tab = metadata['active_tab']
        self.n_parameters = metadata['n_parameters']

    def train(self):
        self.active = False
        process = BackgroundProcess(self._train, (), name=f"{self.title} training")
        self.background_process = process
        process.start()

    def _train(self, process):
        try:
            # get box size for the selected data
            train_data = tifffile.imread(self.train_data_path)
            train_x = train_data[:, 0, :, :, None]
            train_y = train_data[:, 1, :, :, None]
            n_samples = train_x.shape[0]
            box_size = train_x.shape[1]
            # compile, if not compiled yet
            if not self.compiled:
                self.compile(box_size)
            # if training data box size is not compatible with the compiled model, abort.
            if box_size != self.box_size:
                self.train_data_path = f"DATA HAS WRONG BOX SIZE ({box_size[0]} x {box_size[1]})"
                process.set_progress(1.0)
                return

            # train
            self.model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size,
                           callbacks=[TrainingProgressCallback(process, n_samples, self.batch_size),
                                      StopTrainingCallback(process.stop_request)])
        except Exception as e:
            cfg.set_error(e, "Could not train model - see details below.")
            process.stop()

    def compile(self, box_size):
        model_module_name = SEModel.AVAILABLE_MODELS[self.model_enum]
        self.model = SEModel.MODELS[model_module_name]((box_size, box_size, 1))
        self.compiled = True
        self.box_size = box_size
        self.n_parameters = self.model.count_params()

    def set_slice(self, slice_data):
        if not self.compiled:
            return
        if not self.active:
            return
        self.data = self.apply_to_slice(slice_data)
        self.texture.update(self.data)

    def apply_to_slice(self, image):
        # TODO: resize the slice to the model's apix!
        w, h = image.shape
        self.overlap = min([0.67, self.overlap])
        pad_w = self.box_size - (w % self.box_size)
        pad_h = self.box_size - (h % self.box_size)
        # tile
        stride = int(self.box_size * (1.0 - self.overlap))
        boxes = list()
        image = np.pad(image, ((0, pad_w), (0, pad_h)))
        for x in range(0, w - self.box_size + 1, stride):
            for y in range(0, h - self.box_size + 1, stride):
                boxes.append(image[x:x + self.box_size, y:y + self.box_size])
        boxes = np.array(boxes)

        # apply model
        segmentations = np.squeeze(self.model.predict(boxes))

        # detile
        outimage = np.zeros((w + pad_w, h + pad_h))
        count = np.zeros((w + pad_w, h + pad_h), dtype=int)
        i = 0
        for x in range(0, w - self.box_size + 1, stride):
            for y in range(0, h - self.box_size + 1, stride):
                outimage[x:x + self.box_size, y:y + self.box_size] += segmentations[i]
                count[x:x + self.box_size, y:y + self.box_size] += 1
                i += 1

        count[count == 0] = 1
        outimage = outimage / count
        return outimage[:w, :h]

    def apply_to_tomogram(self, path_or_volume):
        # todo: resize the slices to the model's apix!
        if isinstance(path_or_volume, str):
            data = mrcfile.mmap(path_or_volume, mode='r').data
        else:
            data = path_or_volume
        d, w, h = data.shape
        self.overlap = min([0.67, self.overlap])
        pad_w = self.box_size - (w % self.box_size)
        pad_h = self.box_size - (h % self.box_size)

        # tile
        stride = int(self.box_size * (1.0 - self.overlap))
        boxes = list()
        for z in range(d):
            slice = np.pad(data[z, :, :], ((0, pad_w), (0, pad_h)))
            for x in range(0, w - self.box_size + 1, stride):
                for y in range(0, h - self.box_size + 1, stride):
                    boxes.append(slice[x:x+self.box_size, y:y+self.box_size])
        boxes = np.array(boxes)

        # apply model
        segmentations = np.squeeze(self.model.predict(boxes))

        # detile
        volume = np.zeros((d, w + pad_w, h + pad_h))
        count = np.zeros((d, w + pad_w, h + pad_h), dtype=int)
        i = 0
        for z in range(d):
            for x in range(0, w - self.box_size + 1, stride):
                for y in range(0, h - self.box_size + 1, stride):
                    volume[z, x:x+self.box_size, y:y+self.box_size] += segmentations[i]
                    count[z, x:x+self.box_size, y:y+self.box_size] += 1
                    i += 1
        count[count == 0] = 1
        volume = volume / count
        return volume[:, :w, :h]

    @staticmethod
    def load_models():
        model_files = glob.glob(os.path.join(cfg.root, "models", "*.py"))
        for file in model_files:
            try:
                module_name = os.path.basename(file)[:-3]
                mod = importlib.import_module(("scNodes." if not cfg.frozen else "")+"models."+module_name)
                SEModel.MODELS[mod.title] = mod.create
            except Exception as e:
                cfg.set_error(e, "Could not load SegmentationEditor model at path: "+file)
        SEModel.MODELS_LOADED = True
        SEModel.AVAILABLE_MODELS = list(SEModel.MODELS.keys())

    def __eq__(self, other):
        if isinstance(other, SEModel):
            return self.uid == other.uid
        return False


class TrainingProgressCallback(Callback):
    def __init__(self, process, n_samples, batch_size):
        super().__init__()
        self.process = process
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.batches_in_epoch = 0
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.batches_in_epoch = self.n_samples // self.batch_size
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        progress_in_current_epoch = (batch + 1) / self.batches_in_epoch
        total_progress = (self.current_epoch + progress_in_current_epoch) / self.params['epochs']
        self.process.set_progress(total_progress)


class StopTrainingCallback(Callback):
    def __init__(self, stop_request):
        super().__init__()
        self.stop_request = stop_request

    def on_batch_end(self, batch, logs=None):
        if self.stop_request.is_set():
            self.model.stop_training = True


class BackgroundProcess:
    idgen = count(0)

    def __init__(self, function, args, name=None):
        self.uid = next(BackgroundProcess.idgen)
        self.function = function
        self.args = args
        self.name = name
        self.thread = None
        self.progress = 0.0
        self.stop_request = threading.Event()

    def start(self):
        _name = f"BackgroundProcess {self.uid} - "+(self.name if self.name is not None else "")
        self.thread = threading.Thread(daemon=True, target=self._run, name=_name)
        self.thread.start()

    def _run(self):
        self.function(*self.args, self)

    def set_progress(self, progress):
        self.progress = progress

    def stop(self):
        self.stop_request.set()
        self.progress = 1.0

    def __str__(self):
        return f"BackgroundProcess {self.uid} with function {self.function} and args {self.args}"

