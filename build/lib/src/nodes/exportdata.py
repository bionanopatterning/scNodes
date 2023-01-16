from scNodes.node import *
from tkinter import filedialog
from scNodes import util
import os
from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")


def create():
    return ExportDataNode()


class ExportDataNode(Node):
    title = "Export data"
    group = "Data IO"
    colour = (138 / 255, 8 / 255, 8 / 255, 1.0)
    sortid = 5

    def __init__(self):
        super().__init__()
        self.size = 210

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_RECONSTRUCTION])
        self.dataset_in.title = "Any"

        self.path = "..."
        self.roi = [0, 0, 0, 0]
        self.include_discarded_frames = False
        self.saving = False
        self.export_type = 0  # 0 for dataset, 1 for image.
        self.frames_to_load = list()
        self.n_frames_to_save = 1
        self.n_frames_saved = 0

        self.parallel = True
        self.returns_image = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_in.render_end()

            self.export_type = -1
            if self.dataset_in.get_incoming_attribute_type() == ConnectableAttribute.TYPE_DATASET:
                self.export_type = 0
            elif self.dataset_in.get_incoming_attribute_type() == ConnectableAttribute.TYPE_IMAGE:
                self.export_type = 1
            elif self.dataset_in.get_incoming_attribute_type() == ConnectableAttribute.TYPE_RECONSTRUCTION:
                self.export_type = 2
            else:
                self.dataset_in.title = "Any"
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if self.export_type in [0, 1]:
                _, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            imgui.text("Output path")
            imgui.push_item_width(150)
            _, self.path = imgui.input_text("##intxt", self.path, 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                filename = filedialog.asksaveasfilename()
                if filename is not None:
                    if '.' in filename[-5:]:
                        filename = filename[:filename.rfind(".")]
                    self.path = filename
            if self.export_type == 0:
                _c, self.parallel = imgui.checkbox("Parallel", self.parallel)
            content_width = imgui.get_window_width()
            save_button_width = 85
            save_button_height = 25

            if self.saving:
                imgui.spacing()
                imgui.text("Export progress")
                self.progress_bar(self.n_frames_saved / self.n_frames_to_save)

            imgui.spacing()
            imgui.spacing()
            imgui.spacing()
            imgui.new_line()
            imgui.same_line(position=(content_width - save_button_width) // 2)
            if not self.saving:
                if imgui.button("Save", save_button_width, save_button_height):
                    self.do_save()
            else:
                if imgui.button("Cancel", save_button_width, save_button_height):
                    self.saving = False
            super().render_end()

    def get_img_and_save(self, idx):
        img_pxd = self.get_image_impl(idx)
        if self.use_roi:
            img_pxd = img_pxd[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        Image.fromarray(img_pxd).save(self.path+"/0"+str(idx)+".tif")

    def do_save(self):
        if cfg.profiling:
            time_start = time.time()
        if self.export_type == 0:  # Save stack
            self.saving = True
            #if self.include_discarded_frames:
                #n_active_frames = Node.get_source_load_data_node(self).dataset.n_frames
                #self.frames_to_load = list(range(0, n_active_frames))

            self.frames_to_load = list()
            for i in range(Node.get_source_load_data_node(self).dataset.n_frames):
                self.frames_to_load.append(i)
            self.n_frames_to_save = len(self.frames_to_load)
            self.n_frames_saved = 0
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
        elif self.export_type == 1:  # Save image
            img = self.dataset_in.get_incoming_node().get_image(idx=None)
            img_pxd = img.load()
            if self.use_roi:
                img_pxd = img_pxd[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
            try:
                util.save_tiff(img_pxd, self.path + ".tif", pixel_size_nm=img.pixel_size)
            except Exception as e:
                cfg.set_error(e, "Error saving image: "+str(e))
        elif self.export_type == 2: # Save particle data
            try:
                self.dataset_in.get_incoming_node().get_particle_data().save_as_csv(self.path+".csv")
            except Exception as e:
                cfg.set_error(e, "Error saving .csv\n"+str(e))
        if cfg.profiling:
            self.profiler_time += time.time() - time_start

    def on_update(self):
        if self.saving:
            try:
                if self.parallel:
                    indices = list()
                    for i in range(min([cfg.batch_size, len(self.frames_to_load)])):
                        self.n_frames_saved += 1
                        indices.append(self.frames_to_load[-1])
                        self.frames_to_load.pop()
                    Parallel(n_jobs=cfg.batch_size, mmap_mode=settings.joblib_mmmode)(delayed(self.get_img_and_save)(index) for index in indices)
                else:
                    self.n_frames_saved += 1
                    self.get_img_and_save(self.frames_to_load[-1])
                    self.frames_to_load.pop()

                if len(self.frames_to_load) == 0:
                    self.saving = False
            except Exception as e:
                self.saving = False
                cfg.set_error(e, "Error saving stack: \n"+str(e))

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            incoming_img = data_source.get_image(idx)
            img_pxd = incoming_img.load()
            incoming_img.clean()
            return img_pxd
