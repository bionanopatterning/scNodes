from scNodes.core.node import *
from tkinter import filedialog
from scNodes.core.util import get_filetype


def create():
    return LoadDataNode()


class LoadDataNode(Node):
    title = "Import dataset"
    group = "Data IO"

    colour = (84 / 255, 77 / 255, 222 / 255, 1.0)

    sortid = 0
    IMPORT_BATCH_SIZE = 10
    
    def __init__(self):
        super().__init__()
        self.size = 200

        # Set up connectable attributes
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)

        # flags
        self.NODE_IS_DATA_SOURCE = True

        # Set up node-specific vars
        self.dataset = Dataset()
        self.params["path"] = ""
        self.params["pixel_size"] = 64.0
        self.pixel_size = self.params["pixel_size"]
        self.params["load_on_the_fly"] = True
        self.done_loading = False
        self.to_load_idx = 0
        self.n_to_load = 1

        self.params["file_filter_positive_raw"] = ""
        self.params["file_filter_negative_raw"] = ""

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_out"].render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text("Select source file")
            imgui.push_item_width(150)
            _, self.params["path"] = imgui.input_text("##intxt", self.params["path"], 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                selected_file = filedialog.askopenfilename()
                if selected_file is not None:
                    if get_filetype(selected_file) in ['.tiff', '.tif']:
                        self.params["path"] = selected_file
                        self.on_select_file()
            imgui.columns(2, border = False)
            imgui.text("frames:")
            imgui.text("image size:")
            imgui.text("pixel size:  ")
            imgui.next_column()
            imgui.new_line()
            imgui.same_line(spacing=3)
            imgui.text(f"{self.dataset.n_frames}")
            imgui.new_line()
            imgui.same_line(spacing=3)
            imgui.text(f"{self.dataset.img_width}x{self.dataset.img_height}")
            imgui.push_item_width(45)
            _c, self.params["pixel_size"] = imgui.input_float("##nm", self.params["pixel_size"], 0.0, 0.0, format = "%.1f")
            self.any_change = self.any_change or _c
            if _c:
                self.dataset.pixel_size = self.params["pixel_size"]
                self.pixel_size = self.params["pixel_size"]
            imgui.pop_item_width()
            imgui.same_line()
            imgui.text("nm")
            imgui.columns(1)


            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        _, self.params["load_on_the_fly"] = imgui.checkbox("Load on the fly", self.params["load_on_the_fly"])
        if not self.params["load_on_the_fly"] and not self.done_loading:
            self.progress_bar(min([self.to_load_idx / self.n_to_load]))
            imgui.spacing()
            imgui.spacing()
            imgui.spacing()

        imgui.text("Title must contain:")
        av_width = imgui.get_content_region_available_width()
        imgui.set_next_item_width(av_width)
        _c, self.params["file_filter_positive_raw"] = imgui.input_text("##filt_pos", self.params["file_filter_positive_raw"], 1024, imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        Node.tooltip("Enter tags, e.g. 'GFP;RFP', that must be in a filename\n"
                     "in order to retain the frame. Separate tags by a semicolon.\n"
                     "When multiple tags are entered, frames are retained if \n"
                     "any of the tags is in the filename (not necessarily all).\n"
                     "Leave empty to retain all files by default.")
        imgui.text("Title must not contain:")
        imgui.set_next_item_width(av_width)
        _c, self.params["file_filter_negative_raw"] = imgui.input_text("##filt_neg", self.params["file_filter_negative_raw"], 1024, imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        Node.tooltip("Enter tags, e.g. 'GFP;RFP', to select frames for deletion.\n"
                     "Separate by a semicolon. When the positive and negative\n"
                     "selection criteria contradict, frames are retained.")
        if imgui.button("Filter", av_width / 2 - 5, 25):
            self.dataset.filter_frames_by_title(self.params["file_filter_positive_raw"], self.params["file_filter_negative_raw"])
            self.any_change = True
        imgui.same_line(spacing=10)
        if imgui.button("Reset", av_width / 2 - 5, 25):
            self.on_select_file()
            self.any_change = True

    def on_select_file(self):
        try:
            self.dataset = Dataset(self.params["path"], self.params["pixel_size"])
            self.n_to_load = self.dataset.n_frames
            self.done_loading = False
            self.to_load_idx = 0
            self.any_change = True
            cfg.image_viewer.center_image_requested = True
            cfg.set_active_node(self)
        except Exception as e:
            cfg.set_error(e, f"Error importing '{self.params['path']}' as tif stack. Are you sure the data is .tif and at most 3 dimensional (x, y, z/t)?")

    def get_image_impl(self, idx=None):
        if self.dataset.n_frames > 0:
            retimg = copy.deepcopy(self.dataset.get_indexed_image(idx=None))
            retimg.pixel_size = self.params["pixel_size"]
            retimg.clean()
            return retimg
        else:
            return None

    def on_update(self):
        if not self.params["load_on_the_fly"] and not self.done_loading:
            if cfg.profiling:
                time_start = time.time()
                self.profiler_count += 1
            if self.to_load_idx < self.dataset.n_frames:
                self.dataset.get_indexed_image(self.to_load_idx).load()
                self.to_load_idx += 1
                for i in range(LoadDataNode.IMPORT_BATCH_SIZE):
                    if self.to_load_idx < self.dataset.n_frames:
                        self.dataset.get_indexed_image(self.to_load_idx).load()
                        self.to_load_idx += 1
            else:
                self.done_loading = True
            if cfg.profiling:
                self.profiler_time += time.time() - time_start

    def pre_pickle_impl(self):
        cfg.pickle_temp["dataset"] = self.dataset
        self.dataset = Dataset()

    def post_pickle_impl(self):
        self.dataset = cfg.pickle_temp["dataset"]

    def on_load(self):
        self.on_select_file()

