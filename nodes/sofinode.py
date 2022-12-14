from node import *
from nodes.pysofi import pysofi
from nodes.pysofi import finterp
import os
import tifffile

def create():
    return SofiNode()


class SofiNode(Node):
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
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent=self)

        self.result = None
        self.range_option = 1
        self.range_min = 0
        self.range_max = 100
        self.output_option = 1
        self.moment_order = 4
        self.cumulant_order = 4
        self.magnification = 2

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
        self.temp_dir = "_srnodes_temp_dir_sofinode_"+str(self.id)

    def render(self):
        if super().render_start():  # as in the above __init__ function, the render function must by calling the base class' render_start() function, and end with a matching render_end() - see below.
            self.dataset_in.render_start()  # calling a ConnectableAttribute's render_start() and render_end() handles the rendering and connection logic for that attribute.
            self.image_out.render_start()
            self.dataset_in.render_end()
            self.image_out.render_end()  # callin start first for both attributes and end afterwards makes the attributes appear on the same line / at the same height.

            imgui.spacing()  # add a small vertical whitespace
            imgui.separator()  # draw a line separating the above connectors from the rest of the body of the node. purely visual.
            imgui.spacing()



            imgui.set_next_item_width(120)
            _c, self.output_option = imgui.combo("Output type", self.output_option, SofiNode.OUTPUT_OPTIONS)
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
            if self.output_option == 0:
                _c, self.moment_order = imgui.input_int("Moment order", self.moment_order, 1, 1)
                self.any_change = self.any_change or _c
            elif self.output_option == 1:
                _c, self.cumulant_order = imgui.input_int("Cumulant order", self.cumulant_order, 1, 1)
                self.any_change = self.any_change or _c

            imgui.set_next_item_width(120)
            _c, self.range_option = imgui.combo("Range", self.range_option, SofiNode.RANGE_OPTIONS)
            self.any_change = self.any_change or _c

            if self.range_option == 1:
                imgui.text("Range:")
                imgui.push_item_width(30)
                _c, self.range_min = imgui.input_int("start", self.range_min, 0, 0)
                self.any_change = self.any_change or _c
                imgui.same_line()
                _c, self.range_max = imgui.input_int("stop", self.range_max, 0, 0)
                self.any_change = self.any_change or _c
                imgui.pop_item_width()

            imgui.pop_item_width()
            imgui.set_next_item_width(60)
            _c, self.magnification = imgui.input_int("Magnification", self.magnification, 0, 0)
            self.magnification = max([1, self.magnification])
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
        if self.range_option == 0:
            dataset_source = Node.get_source_load_data_node(self)
            self.frames_requested = list(range(0, dataset_source.dataset.n_frames))
            self.range_min = 0
        else:
            self.frames_requested = list(range(self.range_min, self.range_max))
        sample_input_img = self.dataset_in.get_incoming_node().get_image_impl(idx=0)
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
            datasource = self.dataset_in.get_incoming_node()
            self.stack[self.frames_requested[0] - self.range_min, :, :] = datasource.get_image(idx = self.frames_requested[0]).load()
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
        if self.output_option == 0:
            self.sofi_img = self.pysd.moment_image(order=self.moment_order, finterp=self.magnification > 1, interp_num=self.magnification)
        elif self.output_option == 1:
            self.cumulant_imgs = self.pysd.cumulants_images(highest_order=self.cumulant_order)
            if self.magnification > 1:
                self.sofi_img = finterp.fourier_interp_array(self.cumulant_imgs[self.cumulant_order], [self.magnification])[0]
            else:
                self.sofi_img = self.cumulant_imgs[self.cumulant_order]
        self.any_change = True
        self.processing = False

    def get_image_impl(self, idx):
        if self.sofi_img is not None:
            outframe = Frame("sofi node output image")
            outframe.data = self.sofi_img
            outframe.pixel_size = self.pxsizein / self.magnification
            return outframe
        else:
            return None
