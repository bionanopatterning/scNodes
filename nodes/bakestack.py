from node import *
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

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, self)
        self.coordinates_in = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.INPUT, self)
        self.coordinates_out = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.OUTPUT, self)

        self.parallel = True
        self.baking = False
        self.n_to_bake = 1
        self.n_baked = 0
        self.range_option = 0
        self.custom_range_min = 0
        self.custom_range_max = 1
        self.has_dataset = False
        self.dataset = Dataset()
        self.frames_to_bake = list()
        self.temp_dir_do_not_edit = "_srnodes_temp_dir_bakestack_"+str(self.id)
        self.joblib_dir_counter = 0
        self.baked_at = 0
        self.has_coordinates = False
        self.coordinates = list()
        self.bake_coordinates = False

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_in.render_end()
            if self.has_dataset:
                self.dataset_out.render_start()
                self.dataset_out.render_end()
            if self.bake_coordinates:
                imgui.spacing()
                self.coordinates_in.render_start()
                self.coordinates_in.render_end()
                if self.has_coordinates:
                    self.coordinates_out.render_start()
                    self.coordinates_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.bake_coordinates = imgui.checkbox("Also bake coordinates", self.bake_coordinates)
            if _c and not self.bake_coordinates:
                self.coordinates_in.disconnect_all()

            if self.range_option == 1:
                imgui.push_item_width(80)
                _c, (self.custom_range_min, self.custom_range_max) = imgui.input_int2('[start, top) index', self.custom_range_min, self.custom_range_max)
                imgui.pop_item_width()
            _c, self.parallel = imgui.checkbox("Parallel processing", self.parallel)
            imgui.set_next_item_width(100)
            _c, self.range_option = imgui.combo("Range to bake", self.range_option, BakeStackNode.RANGE_OPTIONS)
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
            imglist = glob.glob(self.temp_dir_do_not_edit +"/*.tif*")
            for img in imglist:
                shutil.copy(img, fpath)

    def get_image_and_save(self, idx=None):
        datasource = self.dataset_in.get_incoming_node()
        if datasource:
            pxd = datasource.get_image(idx).load()
            Image.fromarray(pxd).save(self.temp_dir_do_not_edit + "/0" + str(idx) + ".tif")
            if self.bake_coordinates:
                coordsource = self.coordinates_in.get_incoming_node()
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
            datasource = self.dataset_in.get_incoming_node()
            if datasource:
                return datasource.get_image(idx)

    def get_coordinates(self, idx=None):
        if self.has_coordinates and idx in range(0, self.dataset.n_frames):
            return self.coordinates[idx]

    def init_bake(self):
        self.dataset = None
        if os.path.isdir(self.temp_dir_do_not_edit):
            for f in glob.glob(self.temp_dir_do_not_edit+"/*"):
                os.remove(f)
        else:
            os.mkdir(self.temp_dir_do_not_edit)
        self.has_dataset = False
        self.has_coordinates = False
        self.coordinates = list()
        if self.range_option == 0:
            dataset_source = Node.get_source_load_data_node(self)
            self.frames_to_bake = list(range(0, dataset_source.dataset.n_frames))
        elif self.range_option == 1:
            self.frames_to_bake = list(range(self.custom_range_min, self.custom_range_max))
        self.n_baked = 0
        self.baked_at = datetime.datetime.now().strftime("%H:%M")
        self.n_to_bake = max([1, len(self.frames_to_bake)])

    def on_update(self):
        if self.baking:
            if cfg.profiling:
                time_start = time.time()
            try:
                if self.parallel:
                    indices = list()
                    for i in range(min([cfg.batch_size, len(self.frames_to_bake)])):
                        self.n_baked += 1
                        indices.append(self.frames_to_bake[-1])
                        self.frames_to_bake.pop()
                    coordinates = Parallel(n_jobs=cfg.batch_size, mmap_mode=settings.joblib_mmmode)(delayed(self.get_image_and_save)(index) for index in indices)
                else:
                    index = self.frames_to_bake[-1]
                    self.frames_to_bake.pop()
                    coordinates = self.get_image_and_save(index)
                    self.n_baked += 1
                    ## todo fix error where bakestack freezes upon leaving entire node setup intact, changing source data node input to a different stack, then pressing GO on bake stack node.
                if self.bake_coordinates:
                    self.coordinates += coordinates  # coordinates is a list of lists. in the end, self.coordinates will be a list of length [amount of frames], with a sublist of xy coords for every img.
                if cfg.profiling:
                    self.profiler_time += time.time() - time_start
                    self.profiler_count += len(indices)

                if len(self.frames_to_bake) == 0:
                    # Load dataset from temp dir.
                    self.dataset = Dataset(self.temp_dir_do_not_edit + "/00.tif")
                    print("Done baking - now loading from disk")
                    for frame in self.dataset.frames:
                        frame.load()
                    print("Done loading")
                    self.any_change = True
                    self.baking = False
                    self.play = False
                    self.has_dataset = True
                    if self.bake_coordinates:
                        self.coordinates.reverse()
                        self.has_coordinates = True
            except Exception as e:
                self.baking = False
                self.play = False
                cfg.set_error(e, "Error baking stack: \n"+str(e))

