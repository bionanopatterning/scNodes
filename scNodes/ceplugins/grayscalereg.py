from scNodes.core.ceplugin import *
from copy import copy, deepcopy
from pystackreg import StackReg
import cv2


def create():
    return GrayscaleRegPlugin()


class GrayscaleRegPlugin(CEPlugin):
    title = "Register Grayscale"
    description = "Register an image (the ' child') onto a (larger) parent image, based on image intensity. The plugin\n" \
                  "assumes that the child is positioned roughly in the right place. If the registration fails, try" \
                  "toggling the 'Normalize' option to filter images before alignment." \
                  "\n" \
                  "Alignment is performed using TurboReg - see the Registration node for a reference."

    REGMODES = [StackReg.TRANSLATION, StackReg.RIGID_BODY, StackReg.SCALED_ROTATION]
    REGMODES_STR = ["T", "T+R", "T+R+S"]


    def __init__(self):
        self.parent_frame = None
        self.child_frame = None
        self.regmode = 0
        self.bin = False
        self.binfac = 2
        self.smooth = False
        self.smoothfac = 2.0
        self.normalize = True
        self.input_pos = True


    def render(self):
        _c, self.parent_frame = self.widget_select_frame_no_rgb("Parent frame:", self.parent_frame)
        imgui.text("Child frame:")
        self.child_frame = CEPlugin.widget_show_active_frame_title()

        imgui.spacing()
        _cw = imgui.get_content_region_available_width()
        imgui.push_item_width(_cw-120)
        _c, self.regmode = imgui.combo("Transform type", self.regmode, GrayscaleRegPlugin.REGMODES_STR)
        imgui.pop_item_width()
        self.tooltip("T = Translation, R = Rotation, S = Scaling.\n"
                     "Note: when allowing Scaling, the resulting image pixel size may differ from the\n"
                     "value found in the original image's header file or what was input by the user.\n"
                     "Mode T or T+R is usually enough - allowing scaling tends to worsens results.\n\n")

        # normalizing, binning and smoothing
        _c, self.normalize = imgui.checkbox("Normalize", self.normalize)
        # _c, self.bin = imgui.checkbox("Bin", self.bin)
        # if self.bin:
        #     imgui.same_line(position=110)
        #     imgui.set_next_item_width(30)
        #     _c, self.binfac = imgui.input_int("factor##b", self.binfac, 0.0, 0.0)
        #     self.binfac = max([1, self.binfac])
        #     self.tooltip("Bin images prior to registration. Factor is the binning factor.")
        # _c, self.smooth = imgui.checkbox("Smooth", self.smooth)
        # if self.smooth:
        #     imgui.same_line(position=110)
        #     imgui.set_next_item_width(30)
        #     _c, self.smoothfac = imgui.input_float("factor##s", self.smoothfac, 0.0, 0.0, format="%.1f")
        #     self.smoothfac = max([0.1, self.smoothfac])
        #     self.tooltip("Apply a Gaussian blur to the images prior to registration.\n"
        #                  "Factor is the stdev (in units of the child image's pixel size) of the kernel")

        imgui.spacing()
        if self.widget_centred_button("Align!"):
            self.align_frames()


    def align_frames(self):
        try:
            # Filter, resize, and crop images
            p = self.filter_image(self.parent_frame.data)
            c = self.filter_image(self.child_frame.data)
            npix_p = self.parent_frame.pixel_size
            npix_c = self.child_frame.pixel_size
            _size = np.asarray(np.shape(c)) * npix_c / npix_p
            _size = _size.astype(np.int16) // 2 * 2
            _child = cv2.resize(c, dsize=(_size[1], _size[0]), interpolation=1)  # interpolation=1 == cv2.INTER_LINEAR
            ## GET PIXEL COORDINATES OF WORLD LOCATION CURSOR
            pixel_offset = [0, 0]
            if self.input_pos:
                pixel_coordinate, pixel_offset = self.parent_frame.world_to_pixel_coordinate(self.child_frame.transform.translation)
                _parent = self.crop_around_coordinate(p, _size, pixel_coordinate)
            else:
                _parent = self.crop_center(p, _size)
            if self.normalize:
                _child -= np.mean(_child)
                _child /= np.mean(np.abs(_child))
                _parent -= np.mean(_parent)
                _parent /= np.mean(np.abs(_parent))
            # if self.bin:
            #     width, height = _child.shape
            #     _child = _child[:self.binfac * (width // self.binfac), :self.binfac * (height // self.binfac)]
            #     _parent = _parent[:self.binfac * (width // self.binfac), :self.binfac * (height // self.binfac)]
            #     _child = _child.reshape((width // self.binfac, self.binfac, height // self.binfac, self.binfac)).mean(3).mean(1)
            #     _parent = _parent.reshape((width // self.binfac, self.binfac, height // self.binfac, self.binfac)).mean(3).mean(1)
            # if self.smooth:
            #     _child = gaussian_filter(_child, self.smoothfac)
            #     _parent = gaussian_filter(_parent, self.smoothfac)


            # Find transformation matrix that matches the frames
            sr = StackReg(GrayscaleRegPlugin.REGMODES[self.regmode])
            tmat = sr.register(_parent, _child)
            regd = sr.transform(_child, tmat)
            T, R, S = self.decompose_transform_matrix(tmat)
            # Apply transform to child
            #self.child_frame.parent_to(self.parent_frame)
            #self.child_frame.transform = deepcopy(self.parent_frame.transform)
            dx = (0.0 * pixel_offset[0] - T[0]) * npix_p
            dy = (0.0 * pixel_offset[1] - T[1]) * npix_p
            self.child_frame.translate([dx, dy])
            self.child_frame.pivoted_rotation([self.child_frame.transform.translation[0], self.child_frame.transform.translation[1]], R)
            self.child_frame.pivoted_scale([self.child_frame.transform.translation[0], self.child_frame.transform.translation[1]], S)
            self.child_frame.parent_to(self.parent_frame)

        except Exception as e:
            cfg.set_error(e, "Error aligning frames in Register Grayscale tool.")


    @staticmethod
    def crop_center(img, size):
        w, h = img.shape
        dw = (w - size[0]) // 2
        dh = (h - size[1]) // 2
        return img[dw:dw + size[0], dh:dh + size[1]]


    @staticmethod
    def crop_around_coordinate(img, size, coordinate):
        W = size[0]
        H = size[1]
        w, h = img.shape
        if W > w or H > h:
            raise Exception("ROI can't be larger than the original image.")
        X = coordinate[1]
        Y = coordinate[0]
        X = max([W // 2, min([X, w - W // 2])])
        Y = max([H // 2, min([Y, h - H // 2])])
        return img[X - W // 2:X + W // 2, Y - H // 2:Y + H // 2]


    @staticmethod
    def filter_image(array):
        data = copy(array)
        mu = np.mean(data)
        std = np.std(data)
        mask = data > (mu + std * 10.0)
        data[mask] = mu
        data = data - np.amin(data)
        data = data / np.amax(data)
        return data


    @staticmethod
    def decompose_transform_matrix(tmat):
        vec = np.matrix(tmat[0:2, 0:2]) * np.matrix([[1], [0]])
        scaling = float(np.sqrt(vec[0] ** 2 + vec[1] ** 2))
        rotation = float(np.arctan2(vec[1], vec[0]) * 360.0 / 2.0 / np.pi)
        translation = np.array([tmat[0, 2], tmat[1, 2]])
        return translation, rotation, scaling
