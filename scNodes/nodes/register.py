from scNodes.core.node import *
from pystackreg import StackReg
import cv2


def create():
    return RegisterNode()


class RegisterNode(Node):
    title = "Registration"
    group = "Image processing"  # default groups: "Data IO", "Image processing", "Reconstruction", "Custom"
    colour = (68 / 255, 177 / 255, 209 / 255, 1.0)
    sortid = 100
    METHODS = ["TurboReg", "ORB"]
    REFERENCES = ["Input image", "Template frame", "Consecutive pairing"]
    INTERPOLATION_OPTIONS = ["Nearest neighbour", "Bilinear", "Biquadratic", "Bicubic", "Biquartic", "Biquintic"]
    EDGE_FILL_OPTIONS = ["Zero", "Repeat", "Reflect"]
    edge_fill_options_scipy_argument = ['constant', 'edge', 'reflect']

    def __init__(self):
        super().__init__()  # Was: super(LoadDataNode, self).__init__()
        self.size = 230

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.image_in = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.INPUT, parent = self)

        # Set up node-specific vars
        self.buffer_last_output = True
        self.register_method = 0
        self.reference_method = 1
        self.reference_image = None
        self.frame = 0
        self.roi = [0, 0, 0, 0]

        # StackReg vars
        self.sr = StackReg(StackReg.TRANSLATION)

        # CV vars
        self.orb = None
        self.orb_n_actual = 0
        self.orb_n_requested = 500
        self.orb_keep = 0.7
        self.orb_confidence = 0.99
        self.orb_method = 0

        # Advanced
        self.interpolation = 1
        self.edge_fill = 0
        self.preserve_range = False

    def render(self):
        ## TODO: refresh input image when input image was updated.
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            self.any_change = self.any_change or _c
            self.tooltip("When 'use ROI' is active, the translation required to register the image is determined based\n"
                         "on the data in the ROI only. This can help speed up registration, as well as avoid errors\n"
                         "such as occur when the registrator non-constant image features like blinking particles are\n"
                         "present in the full image.")
            imgui.push_item_width(140)
            _method_changed, self.register_method = imgui.combo("Method", self.register_method, RegisterNode.METHODS)
            if self.register_method == 0:
                imgui.same_line()
                imgui.button("?", 19, 19)
                self.tooltip("This method uses pyStackReg, a TurboReg wrapper by Gregor Lichtner (@glichtner). Original publication:\nP. Thévenaz, U. E. Ruttimann, M. Unser (1998) A Pyramid Approach to Subpixel Registration Based on Intensity. \nIEEE Trans. Image Process. 7:1:27-41 doi: 10.1109/83.650848")
            _reference_changed, self.reference_method = imgui.combo("Reference", self.reference_method, RegisterNode.REFERENCES)
            imgui.pop_item_width()

            _frame_changed = False
            if self.reference_method == 1:
                imgui.push_item_width(50)
                _frame_changed, self.frame = imgui.input_int("Template frame", self.frame, 0, 0)
                imgui.pop_item_width()
            if self.reference_method == 0:
                imgui.new_line()
                self.image_in.render_start()
                self.image_in.render_end()

            if self.register_method == 0:
                pass  # no options (yet) for TurboReg
            elif self.register_method == 1:
                imgui.push_item_width(110)
                _c, self.orb_method = imgui.combo("Estimator", self.orb_method, ["Random sample consensus", "Least median of squares"])
                self.any_change = self.any_change or _c
                imgui.pop_item_width()
                imgui.push_item_width(90)
                _c, self.orb_n_requested = imgui.input_int("# features", self.orb_n_requested, 100, 100)
                self.any_change = self.any_change or _c
                _c, self.orb_keep = imgui.input_float("% matches to keep", self.orb_keep, format = "%.2f")
                self.any_change = self.any_change or _c
                _c, self.orb_confidence = imgui.input_float("confidence", self.orb_confidence, format = "%.2f")
                self.any_change = self.any_change or _c
                if _c:
                    self.orb_confidence = min([1.0, max([0.1, self.orb_confidence])])
                imgui.pop_item_width()

            self.any_change = self.any_change or _method_changed or _reference_changed or _frame_changed
            if self.any_change:
                self.reference_image = None

            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        imgui.push_item_width(110)
        _c, self.interpolation = imgui.combo("Interpolation", self.interpolation, RegisterNode.INTERPOLATION_OPTIONS)
        self.any_change = self.any_change or _c
        _c, self.edge_fill = imgui.combo("Edges", self.edge_fill, RegisterNode.EDGE_FILL_OPTIONS)
        self.any_change = self.any_change or _c
        imgui.pop_item_width()
        Node.tooltip("Set how to fill pixels falling outside of the original image. Edge: clamp the edge\n"
                     "values, Reflect: reflect the image along the boundaries of the original image.")
        _c, self.preserve_range = imgui.checkbox("Preserve range", self.preserve_range)
        self.any_change = self.any_change or _c
        Node.tooltip("When checked, the intensity range of the output, registered image is fixed as the\n"
                     "same range as that of the original image.")

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            input_img = data_source.get_image(idx)
            if self.reference_image is None:
                # Get reference frame according to specified pairing method
                if self.reference_method == 2:
                    self.reference_image = data_source.get_image(idx - 1).load()
                elif self.reference_method == 0:
                    self.reference_image = self.image_in.get_incoming_node().get_image(idx=None).load()
                elif self.reference_method == 1:
                    self.reference_image = data_source.get_image(self.frame).load()

            # Perform registration according to specified registration method

            if self.reference_image is not None:
                template = self.reference_image
                image = input_img.load()
                if self.use_roi:
                    template = template[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                    image = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                if self.register_method == 0:
                    tmat = self.sr.register(template, image)
                    input_img.translation = [tmat[0][2], tmat[1][2]]
                elif self.register_method == 1:
                    ## make new ORB if necessary
                    if self.orb_n_actual != self.orb_n_requested:
                        self.orb = cv2.ORB_create(nfeatures=self.orb_n_requested)
                    tmat = self._orb_register(template, image, keep=self.orb_keep, confidence=self.orb_confidence,
                                              method=self.orb_method)
                    input_img.translation = tmat
            else:
                cfg.set_error(Exception(), "RegisterNode: reference image was None.")

            if self.reference_method == 2:
                self.reference_image = None
            input_img.bake_transform(interpolation=self.interpolation, edges=RegisterNode.edge_fill_options_scipy_argument[self.edge_fill], preserve_range=self.preserve_range)
            return input_img

    def pre_pickle_impl(self):
        self.reference_image = None

    def _orb_register(self, ref, img, keep=0.7, confidence=0.99, method=0):
        """

        :param ref: reference image
        :param img: image to be registered
        :param keep: the fraction (0.0 to 1.0) of matches to use to estimate the transformation - e.g. 0.1: keep top 10% matches
        :param confidence: confidence parameter of cv2.estimateAffinePartial2d
        :param method: 0 for RANSAC, 1 for LMEDS
        :return: list with two elements: x shift and y shift, in pixels.
        """
        cmin = np.amin(img)
        cmax = np.amax(img)
        ref8 = 255 * (ref - cmin) / (cmax - cmin)
        ref8 = ref8.astype(np.uint8)
        img8 = 255 * (img - cmin) / (cmax - cmin)
        img8 = img8.astype(np.uint8)

        kp1, d1 = self.orb.detectAndCompute(ref8, None)
        kp2, d2 = self.orb.detectAndCompute(img8, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = list(matcher.match(d1, d2))
        matches.sort(key=lambda x: x.distance)
        matches = matches[:int(len(matches) * keep)]
        n_matches = len(matches)

        p1 = np.zeros((n_matches, 2))
        p2 = np.zeros((n_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt
            p2[i, :] = kp2[matches[i].trainIdx].pt

        tmat, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC if method == 0 else cv2.LMEDS,
                                                   confidence=confidence)
        print(tmat)
        return [tmat[0][2], tmat[1][2]]
