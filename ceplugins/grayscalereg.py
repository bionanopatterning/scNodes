from ceplugin import *
from copy import copy, deepcopy
from pystackreg import StackReg
import cv2

def create():
    return GrayscaleRegPlugin()


class GrayscaleRegPlugin(CEPlugin):
    title = "Register Grayscale"
    description = "Register images of equal or different sizes based on image intensity.\n" \
                  "Useful to register high-magnification TEM images to low-magnification\n" \
                  "images of the same region (e.g. to register Exposure/Search frames onto\n" \
                  "Overview images.) Requires manual picking of the approximate location \n" \
                  "of the child image (the image to be transformed) within parent image." \

    REGMODES = [StackReg.TRANSLATION, StackReg.RIGID_BODY, StackReg.SCALED_ROTATION]
    REGMODES_STR = ["T", "T+R", "T+R+S"]

    def __init__(self):
        self.parent_frame = None
        self.child_frame = None
        self.FLAG_SHOW_LOCATION_PICKER = True
        self.regmode = 0
        self.input_pos = True

    def render(self):


        _c, self.parent_frame = self.widget_select_frame_no_rgb("Parent frame:", self.parent_frame)
        _c, self.child_frame = self.widget_select_frame_no_rgb("Child frame:", self.child_frame)

        _cw = imgui.get_content_region_available_width()
        imgui.push_item_width(_cw-120)
        _c, self.regmode = imgui.combo("Transform type", self.regmode, GrayscaleRegPlugin.REGMODES_STR)
        imgui.pop_item_width()
        self.tooltip("T = Translation, R = Rotation, S = Scaling.\n"
                     "Note: when allowing Scaling, the resulting image pixel size may differ from the\n"
                     "value found in the original image's header file or what was input by the user.\n"
                     "Mode T or T+R should be good enough - allowing S tends to worsens results.\n\n"
                     "The selected mode is only applied in the image registration step. In the prior\n"
                     "filtering and rough alignment steps, the input images' (differing) pixel\n"
                     "sizes are always taken in to account.")
        _c, self.input_pos = imgui.checkbox("Location hint:", self.input_pos)
        self.FLAG_SHOW_LOCATION_PICKER = self.input_pos
        if self.input_pos:
            self.info_selected_position()

        imgui.spacing()
        if self.centred_button("Align!"):
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
                pixel_coordinate, pixel_offset = self.parent_frame.world_to_pixel_coordinate(self.selected_position)
                _parent = self.crop_around_coordinate(p, _size, pixel_coordinate)
            else:
                _parent = self.crop_center(p, _size)

            # Find transformation matrix that matches the frames
            sr = StackReg(GrayscaleRegPlugin.REGMODES[self.regmode])
            tmat = sr.register(_parent, _child)
            T, R, S = self.decompose_transform_matrix(tmat)

            # Apply transform to child
            self.child_frame.parent_to(self.parent_frame)
            self.child_frame.transform = deepcopy(self.parent_frame.transform)
            self.child_frame.transform.translation[0] += T[0] * npix_c + pixel_offset[0] * npix_p
            self.child_frame.transform.translation[1] += T[1] * npix_c + pixel_offset[1] * npix_p
            self.child_frame.transform.rotation += R
            self.child_frame.transform.scale *= S

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
        X = coordinate[0]
        Y = coordinate[1]
        X = max([W // 2, min([X, w - W // 2])])
        Y = max([H // 2, min([Y, h - H // 2])])
        return img[X - W // 2:X + W // 2, Y - H // 2:Y + H // 2]

    @staticmethod
    def filter_image(array):
        data = copy(array)
        mu = np.mean(data)
        std = np.std(data)
        mask = data > (mu + std * 3.0)
        data[mask == True] = mu  ## todo array[mask]=mu
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
