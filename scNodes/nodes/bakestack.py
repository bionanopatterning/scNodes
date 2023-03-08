from scNodes.core.node import *
import os
from tkinter import filedialog
import shutil
from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")


def create():
    return BakeStackNode()


class BakeStackNode(Node):
    title = "Bake stack"
    group = "Image processing"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    sortid = 120

    RANGE_OPTIONS = ["All frames", "Custom range"]

    def __init__(self):
        super().__init__()
        self.size = 230
        
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, self)
        self.connectable_attributes["coordinates_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.INPUT, self)
        self.connectable_attributes["coordinates_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.OUTPUT, self)

        self.params["parallel"] = True
        self.baking = False
        self.n_to_bake = 1
        self.n_baked = 0
        self.params["range_option"] = 0
        self.params["custom_range_min"] = 0
        self.params["custom_range_max"] = 1
        self.has_dataset = False
        self.dataset = Dataset()
        self.frames_to_bake = list()
        self.temp_dir = self.gen_temp_dir_name()
        self.joblib_dir_counter = 0
        self.baked_at = 0
        self.has_coordinates = False
        self.coordinates = list()
        self.params["bake_coordinates"] = False
        self.params["load_baked_stack_into_ram"] = False

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            if self.has_dataset:
                self.connectable_attributes["dataset_out"].render_start()
                self.connectable_attributes["dataset_out"].render_end()
            if self.params["bake_coordinates"]:
                imgui.spacing()
                self.connectable_attributes["coordinates_in"].render_start()
                self.connectable_attributes["coordinates_in"].render_end()
                if self.has_coordinates:
                    self.connectable_attributes["coordinates_out"].render_start()
                    self.connectable_attributes["coordinates_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.params["bake_coordinates"] = imgui.checkbox("Also bake coordinates", self.params["bake_coordinates"])
            if _c and not self.params["bake_coordinates"]:
                self.connectable_attributes["coordinates_in"].disconnect_all()

            _c, self.params["parallel"] = imgui.checkbox("Parallel processing", self.params["parallel"])
            imgui.set_next_item_width(100)
            _c, self.params["range_option"] = imgui.combo("Range to bake", self.params["range_option"], BakeStackNode.RANGE_OPTIONS)
            if self.params["range_option"] == 1:
                imgui.push_item_width(80)
                _c, (self.params["custom_range_min"], self.params["custom_range_max"]) = imgui.input_int2('[start, top) index', self.params["custom_range_min"], self.params["custom_range_max"])
                imgui.pop_item_width()
                
            clicked, self.baking = self.play_button()
            if clicked and self.baking:
                self.init_bake()

            if self.baking:
                imgui.text("Baking process:")
                self.progress_bar(self.n_baked / self.n_to_bake)
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()

            if self.has_dataset:
                imgui.text(f"Output data was baked at: {self.baked_at}")

            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()
            super().render_end()

    def render_advanced(self):
        _c, self.params["load_baked_stack_into_ram"] = imgui.checkbox("Keep in RAM", self.params["load_baked_stack_into_ram"])
        self.tooltip("Check this box to automatically load entire the baked stack into RAM. Can speed up downstream\n"
                     "processes a bit, but requires sufficient RAM. If checked but RAM is too low, software can be-\n"
                     "come slow, as the Python garbage collector is forced to try to delete unused data.")
        if self.has_dataset:
            _cw = imgui.get_content_region_available_width()
            imgui.new_line()
            imgui.same_line(spacing = _cw / 2 - 80 / 2)
            if imgui.button("Quick save", 80, 25):
                fpath = filedialog.asksaveasfilename()
                if fpath is not None:
                    if '.' in fpath[-5:]:
                        fpath = fpath[:fpath.rfind(".")]
                # now copy the files from the temp folder to the requested folder
                if not os.path.isdir(fpath):
                    os.mkdir(fpath)
                imglist = glob.glob(self.temp_dir +"/*.tif*")
                for img in imglist:
                    shutil.copy(img, fpath)

    def get_image_and_save(self, idx=None):
        datasource = self.connectable_attributes["dataset_in"].get_incoming_node()
        if datasource:
            pxd = datasource.get_image(idx).load()
            Image.fromarray(pxd).save(self.temp_dir + "/0" + str(idx) + ".tif")
            if self.params["bake_coordinates"]:
                coordsource = self.connectable_attributes["coordinates_in"].get_incoming_node()
                coordinates = coordsource.get_coordinates(idx)
                return coordinates
            return False
        return False

    def get_image_impl(self, idx=None):
        if self.has_dataset and idx in range(0, self.dataset.n_frames):
            retimg = copy.deepcopy(self.dataset.get_indexed_image(idx))
            retimg.clean()
            if self.has_coordinates:
                retimg.maxima = self.coordinates[idx]
            return retimg
        else:
            datasource = self.connectable_attributes["dataset_in"].get_incoming_node()
            if datasource:
                return datasource.get_image(idx)

    def get_coordinates(self, idx=None):
        if self.has_coordinates and idx in range(0, self.dataset.n_frames):
            return self.coordinates[idx]

    def init_bake(self):
        del self.dataset
        self.dataset = None
        self.temp_dir = self.gen_temp_dir_name()
        if os.path.isdir(self.temp_dir):
            for f in glob.glob(self.temp_dir+"/*"):
                os.remove(f)
        else:
            os.mkdir(self.temp_dir)
        self.has_dataset = False
        self.has_coordinates = False
        self.coordinates = list()
        if self.params["range_option"] == 0:
            dataset_source = Node.get_source_load_data_node(self)
            self.frames_to_bake = list(range(0, dataset_source.dataset.n_frames))
        elif self.params["range_option"] == 1:
            self.frames_to_bake = list(range(self.params["custom_range_min"], self.params["custom_range_max"]))
        self.n_baked = 0
        self.baked_at = datetime.datetime.now().strftime("%H:%M")
        self.n_to_bake = max([1, len(self.frames_to_bake)])

    def on_update(self):
        if self.baking:
            if cfg.profiling:
                time_start = time.time()
            try:
                if self.params["parallel"]:
                    indices = list()
                    for i in range(min([cfg.batch_size, len(self.frames_to_bake)])):
                        self.n_baked += 1
                        indices.append(self.frames_to_bake[-1])
                        self.frames_to_bake.pop()
                    coordinates = self.parallel_process(self.get_image_and_save, indices)
                else:
                    index = self.frames_to_bake[-1]
                    self.frames_to_bake.pop()
                    coordinates = [self.get_image_and_save(index)]
                    self.n_baked += 1
                if self.params["bake_coordinates"]:
                    self.coordinates += coordinates  # coordinates is a list of lists. in the end, self.coordinates will be a list of length [amount of frames], with a sublist of xy coords for every img.
                if cfg.profiling:
                    self.profiler_time += time.time() - time_start
                    self.profiler_count += len(indices)

                if len(self.frames_to_bake) == 0:
                    # Load dataset from temp dir.
                    self.dataset = Dataset(self.temp_dir + "/00.tif")
                    if self.params["load_baked_stack_into_ram"]:
                        print("Loading baked stack into RAM")
                        for frame in self.dataset.frames:
                            frame.load()
                        print("Done loading baked stack into RAM")
                    self.any_change = True
                    self.baking = False
                    self.play = False
                    self.has_dataset = True
                    if self.params["bake_coordinates"]:
                        self.coordinates.reverse()
                        self.has_coordinates = True
            except Exception as e:
                self.baking = False
                self.play = False
                cfg.set_error(e, "Error baking stack: \n"+str(e))

    def pre_pickle_impl(self):
        cfg.pickle_temp["dataset"] = self.dataset
        self.dataset = Dataset()

    def post_pickle_impl(self):
        self.dataset = cfg.pickle_temp["dataset"]
