from scNodes.core.node import *
from scNodes.nodes.pysofi import pysofi
from scNodes.nodes.pysofi import finterp
import os
import tifffile

def create():
    return SofiNode()


class SofiNode(Node):
    description = "SOFI is a super-resolution reconstruction method that is often used as an alternative to PSF-fitting\n" \
                  "methods when images contain many overlapping fluorescent particles. The SOFI reconstruction node is \n" \
                  "based on Miao et al.'s 'PySOFI: an open source Python package for SOFI.' (2022). The node processes \n" \
                  "the input data in one go, which can cause the software to become unresponsive for a while."
    title = "SOFI reconstruction"
    group = ["Alternative reconstruction"]
    colour = (98 / 255, 13 / 255, 230 / 255, 1.0)  # (230 / 255, 98 / 255, 13 / 255, 1.0)
    size = 245
    sortid = 1006

    RANGE_OPTIONS = ["Full stack", "Selected range"]
    OUTPUT_OPTIONS = ["Moments", "Cumulants"]

    def __init__(self):
        super().__init__()

        # defining in- and output attributes.
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["image_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent=self)

        self.result = None
        self.params["range_option"] = 1
        self.params["range_min"] = 0
        self.params["range_max"] = 100
        self.params["output_option"] = 1
        self.params["moment_order"] = 4
        self.params["cumulant_order"] = 4
        self.params["magnification"] = 2

        self.lut = "Heatmap"
        self.pysd = None
        self.pysd_current = False
        self.cumulant_imgs = dict()
        self.cumulants_up_to_date = False
        self.prev_cumulant_settings = ""
        self.sofi_img = None
        self.stack = None
        self.frames_requested = list()
        self.n_frames_processed = 0
        self.n_frames_requested = 1
        self.pxsizein = 100
        self.stack_prepared = False
        self.processing = False
        self.temp_dir = self.gen_temp_dir_name()

    def render(self):
        if super().render_start():  # as in the above __init__ function, the render function must by calling the base class' render_start() function, and end with a matching render_end() - see below.
            self.connectable_attributes["dataset_in"].render_start()  # calling a ConnectableAttribute's render_start() and render_end() handles the rendering and connection logic for that attribute.
            self.connectable_attributes["image_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["image_out"].render_end()  # callin start first for both attributes and end afterwards makes the attributes appear on the same line / at the same height.

            imgui.spacing()  # add a small vertical whitespace
            imgui.separator()  # draw a line separating the above connectors from the rest of the body of the node. purely visual.
            imgui.spacing()



            imgui.set_next_item_width(120)
            _c, self.params["output_option"] = imgui.combo("Output type", self.params["output_option"], SofiNode.OUTPUT_OPTIONS)
            imgui.same_line()
            imgui.button("?", width=20, height=20)
            self.tooltip("This node is based on PySOFI, see:\n"
                         "Miao et al. 'PySOFI: an open source Python package for SOFI.' (2022)\n\n"
                         ""
                         "During PySOFI processing, GUI will be unresponsive - processing time\n"
                         "varies depending on size of the dataset. Crop images or reduce range\n"
                         "of frames to reduce processing time.")
            imgui.push_item_width(120)
            self.any_change = self.any_change or _c
            if self.params["output_option"] == 0:
                _c, self.params["moment_order"] = imgui.input_int("Moment order", self.params["moment_order"], 1, 1)
                self.any_change = self.any_change or _c
            elif self.params["output_option"] == 1:
                _c, self.params["cumulant_order"] = imgui.input_int("Cumulant order", self.params["cumulant_order"], 1, 1)
                self.any_change = self.any_change or _c

            imgui.set_next_item_width(120)
            _c, self.params["range_option"] = imgui.combo("Range", self.params["range_option"], SofiNode.RANGE_OPTIONS)
            self.any_change = self.any_change or _c

            if self.params["range_option"] == 1:
                imgui.text("Range:")
                imgui.push_item_width(30)
                _c, self.params["range_min"] = imgui.input_int("start", self.params["range_min"], 0, 0)
                self.any_change = self.any_change or _c
                imgui.same_line()
                _c, self.params["range_max"] = imgui.input_int("stop", self.params["range_max"], 0, 0)
                self.any_change = self.any_change or _c
                imgui.pop_item_width()

            imgui.pop_item_width()
            imgui.set_next_item_width(60)
            _c, self.params["magnification"] = imgui.input_int("Magnification", self.params["magnification"], 0, 0)
            self.params["magnification"] = max([1, self.params["magnification"]])
            self.any_change = _c or self.any_change

            if self.processing:
                imgui.text("Preparing stack")
                self.progress_bar(self.n_frames_processed / self.n_frames_requested)
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()

            _cw = imgui.get_content_region_available_width()
            imgui.new_line()
            imgui.same_line(position=(_cw - 100) / 2)
            if imgui.button("Reconstruct", width=100, height=30):
                self.init_reconstruction()
            super().render_end()

    def init_reconstruction(self):
        # unfortunately pysofi works on files saved on disk, not in RAM. numpy arrays as input would be more efficient, but
        # that would require editing the pysofi source which want to avoid. So the first step in building a sofi reconstruction
        # is to get the required data as a numpy array, and save it to disk. This is a bit inefficient.

        # If the current stack range settings are the same as before, skip grabbing the stack.
        self.pysd_current = False

        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)
        if self.params["range_option"] == 0:
            dataset_source = Node.get_source_load_data_node(self)
            self.frames_requested = list(range(0, dataset_source.dataset.n_frames))
            self.params["range_min"] = 0
        else:
            self.frames_requested = list(range(self.params["range_min"], self.params["range_max"]))
        sample_input_img = self.connectable_attributes["dataset_in"].get_incoming_node().get_image_impl(idx=0)
        width, height = sample_input_img.load().shape
        self.pxsizein = sample_input_img.pixel_size
        self.stack = np.zeros((len(self.frames_requested), width, height))
        self.frames_requested.reverse()
        self.stack_prepared = False
        self.processing = True
        self.n_frames_requested = len(self.frames_requested)
        self.n_frames_processed = 0

    def on_update(self):
        if self.processing and not self.stack_prepared:
            datasource = self.connectable_attributes["dataset_in"].get_incoming_node()
            self.stack[self.frames_requested[0] - self.params["range_min"], :, :] = datasource.get_image(idx = self.frames_requested[0]).load()
            self.frames_requested.pop()
            self.n_frames_processed += 1
            if not self.frames_requested:
                self.stack_prepared = True
                # save stack:
                with tifffile.TiffWriter(self.temp_dir + "/sofistack.tif") as stack_out:
                    for i in range(self.stack.shape[0]):
                        stack_out.write(self.stack[i])
                # then prep the reconstruction
                self.pysd = pysofi.PysofiData(self.temp_dir, "sofistack.tif")
                self.pysd_current = True
                self.build_reconstruction()

    def build_reconstruction(self):
        if self.params["output_option"] == 0:
            self.sofi_img = self.pysd.moment_image(order=self.params["moment_order"], finterp=self.params["magnification"] > 1, interp_num=self.params["magnification"])
        elif self.params["output_option"] == 1:
            self.cumulant_imgs = self.pysd.cumulants_images(highest_order=self.params["cumulant_order"])
            if self.params["magnification"] > 1:
                self.sofi_img = finterp.fourier_interp_array(self.cumulant_imgs[self.params["cumulant_order"]], [self.params["magnification"]])[0]
            else:
                self.sofi_img = self.cumulant_imgs[self.params["cumulant_order"]]
        self.any_change = True
        self.processing = False

    def get_image_impl(self, idx):
        if self.sofi_img is not None:
            outframe = Frame("sofi node output image")
            outframe.data = self.sofi_img
            outframe.pixel_size = self.pxsizein / self.params["magnification"]
            return outframe
        else:
            return None
