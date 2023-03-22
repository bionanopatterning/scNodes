from scNodes.core.node import *
from pystackreg import StackReg
import cv2
import pyGPUreg

def create():
    return RegisterNode()


class RegisterNode(Node):
    title = "Registration"
    group = "Image processing"  # default groups: "Data IO", "Image processing", "Reconstruction", "Custom"
    colour = (68 / 255, 177 / 255, 209 / 255, 1.0)
    sortid = 100
    METHODS = ["TurboReg", "ORB", "pyGPUreg"]
    REFERENCES = ["Input image", "Template frame", "Consecutive pairing"]
    INTERPOLATION_OPTIONS = ["Linear", "Cubic"]
    EDGE_FILL_OPTIONS = ["Zero", "Repeat", "Reflect"]
    edge_fill_options_scipy_argument = ['constant', 'edge', 'reflect']

    def __init__(self):
        super().__init__()
        self.size = 230

        # Set up connectable attributes
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes["image_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.INPUT, parent = self)

        # Set up node-specific vars
        self.buffer_last_output = True
        self.params["register_method"] = 0
        self.params["reference_method"] = 1
        self.reference_image = None
        self.params["frame"] = 0
        self.roi = [0, 0, 0, 0]

        # StackReg vars
        self.sr = StackReg(StackReg.TRANSLATION)

        # CV vars
        self.orb = None
        self.orb_n_actual = 0
        self.params["orb_n_requested"] = 500
        self.params["orb_keep"] = 0.7
        self.params["orb_confidence"] = 0.99
        self.params["orb_method"] = 0

        # pyGPUreg vars
        self.params["pyGPUreg_size"] = 256
        self.pygpureg_template_image_set = False
        self.pygpureg_initialized = False
        # Advanced
        self.params["interpolation"] = 1
        self.params["edge_fill"] = 1
        self.params["preserve_range"] = False
        self.params["add_drift_metrics"] = True
        self.params["metric_nm_or_px"] = 0

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_end()
            self.connectable_attributes["dataset_in"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if self.params["register_method"] != 2:
                _c, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
                self.any_change = self.any_change or _c
                self.tooltip("When 'use ROI' is active, the translation required to register the image is determined based\n"
                             "on the data in the ROI only. This can help speed up registration, as well as avoid errors\n"
                             "such as occur when non-constant image features like blinking particles are prevalent in the\n"
                             "full image.")
            else:
                if not self.pygpureg_initialized:
                    pyGPUreg.init(create_window=False, image_size=256)
                    self.pygpureg_initialized = True
                    out, shift = pyGPUreg.register(np.random.uniform(0, 100, (256, 256)), np.random.uniform(0, 100, (256, 256)))
                    print(shift)
                self.use_roi = True
            imgui.push_item_width(140)
            _method_changed, self.params["register_method"] = imgui.combo("Method", self.params["register_method"], RegisterNode.METHODS)
            if self.params["register_method"] == 0:
                imgui.same_line()
                imgui.button("?", 19, 19)
                self.tooltip("This method uses pyStackReg, a TurboReg wrapper by Gregor Lichtner (@glichtner). Original publication:\nP. Thévenaz, U. E. Ruttimann, M. Unser (1998) A Pyramid Approach to Subpixel Registration Based on Intensity. \nIEEE Trans. Image Process. 7:1:27-41 doi: 10.1109/83.650848")
            elif self.params["register_method"] == 2:
                imgui.same_line()
                imgui.button("?", 19, 19)
                self.tooltip("This method uses pyGPUreg, our implementation of GPU-accelerated registration by phase correlation.\nMore information can be found at: github.com/bionanopatterning/pyGPUreg\nNote that pyGPUreg is not compatible with parallel processing on the CPU.")
            _reference_changed, self.params["reference_method"] = imgui.combo("Reference", self.params["reference_method"], RegisterNode.REFERENCES)
            imgui.pop_item_width()

            _frame_changed = False
            if self.params["reference_method"] == 1:
                imgui.push_item_width(50)
                _frame_changed, self.params["frame"] = imgui.input_int("Template frame", self.params["frame"], 0, 0)
                imgui.pop_item_width()
            if self.params["reference_method"] == 0:
                imgui.new_line()
                self.connectable_attributes["image_in"].render_start()
                self.connectable_attributes["image_in"].render_end()

            if self.params["register_method"] == 0:
                pass  # no options (yet) for TurboReg
            elif self.params["register_method"] == 1:
                imgui.push_item_width(110)
                _c, self.params["orb_method"] = imgui.combo("Estimator", self.params["orb_method"], ["Random sample consensus", "Least median of squares"])
                self.any_change = self.any_change or _c
                imgui.pop_item_width()
                imgui.push_item_width(90)
                _c, self.params["orb_n_requested"] = imgui.input_int("# features", self.params["orb_n_requested"], 100, 100)
                self.any_change = self.any_change or _c
                _c, self.params["orb_keep"] = imgui.input_float("% matches to keep", self.params["orb_keep"], format = "%.2f")
                self.any_change = self.any_change or _c
                _c, self.params["orb_confidence"] = imgui.input_float("confidence", self.params["orb_confidence"], format = "%.2f")
                self.any_change = self.any_change or _c
                if _c:
                    self.params["orb_confidence"] = min([1.0, max([0.1, self.params["orb_confidence"]])])
                imgui.pop_item_width()
            elif self.params["register_method"] == 2:
                imgui.push_item_width(95)
                original_size = self.params["pyGPUreg_size"]
                _c, new_size = imgui.input_int("Sample size", self.params["pyGPUreg_size"], 1, 1)
                self.any_change = self.any_change or _c
                self.tooltip("pyGPUreg requires input images to be square and to have a size equal to a power of 2. The ROI is\n"
                            "automatically adjusted, but you can specify how large it should be.")
                if _c:
                    self.pygpureg_template_image_set = False
                    if new_size == original_size-1:
                        self.params["pyGPUreg_size"] //= 2
                    elif new_size == original_size+1:
                        self.params["pyGPUreg_size"] *= 2

            self.any_change = self.any_change or _method_changed or _reference_changed or _frame_changed

            if self.any_change:
                self.reference_image = None
                self.pygpureg_template_image_set = False

            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        imgui.push_item_width(110)
        _c, self.params["interpolation"] = imgui.combo("Interpolation", self.params["interpolation"], RegisterNode.INTERPOLATION_OPTIONS)
        self.any_change = self.any_change or _c
        _c, self.params["edge_fill"] = imgui.combo("Edges", self.params["edge_fill"], RegisterNode.EDGE_FILL_OPTIONS)
        self.any_change = self.any_change or _c
        imgui.pop_item_width()
        Node.tooltip("Set how to fill pixels falling outside of the original image. Edge: clamp the edge\n"
                     "values, Reflect: reflect the image along the boundaries of the original image.")
        _c, self.params["preserve_range"] = imgui.checkbox("Preserve range", self.params["preserve_range"])
        self.any_change = self.any_change or _c
        Node.tooltip("When checked, the intensity range of the output, registered image is fixed as the\n"
                     "same range as that of the original image.")
        _c, self.params["add_drift_metrics"] = imgui.checkbox("Add drift metrics", self.params["add_drift_metrics"])
        if self.params["add_drift_metrics"]:
            imgui.set_next_item_width(110)
            _c, self.params["metric_nm_or_px"] = imgui.combo("Metric unit", self.params["metric_nm_or_px"], ["nm", "pixel"])


    def get_image_impl(self, idx=None):
        data_source = self.connectable_attributes["dataset_in"].get_incoming_node()
        if data_source:
            input_img = data_source.get_image(idx)
            if self.params["register_method"] in [0, 1]:
                if self.reference_image is None:
                    # Get reference frame according to specified pairing method
                    if self.params["reference_method"] == 2:
                        self.reference_image = data_source.get_image(idx - 1).load()
                    elif self.params["reference_method"] == 0:
                        self.reference_image = self.connectable_attributes["image_in"].get_incoming_node().get_image(idx=None).load()
                    elif self.params["reference_method"] == 1:
                        self.reference_image = data_source.get_image(self.params["frame"]).load()

                # Perform registration according to specified registration method
                if self.reference_image is not None:
                    template = self.reference_image
                    image = input_img.load()
                    if self.use_roi:
                        template = template[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                        image = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                    if self.params["register_method"] == 0:
                        tmat = self.sr.register(template, image)
                        input_img.translation = [tmat[0][2], tmat[1][2]]
                    elif self.params["register_method"] == 1:
                        ## make new ORB if necessary
                        if self.orb_n_actual != self.params["orb_n_requested"]:
                            self.orb = cv2.ORB_create(nfeatures=self.params["orb_n_requested"])
                        tmat = self._orb_register(template, image, keep=self.params["orb_keep"], confidence=self.params["orb_confidence"],
                                                  method=self.params["orb_method"])
                        input_img.translation = tmat
                else:
                    cfg.set_error(Exception(), "RegisterNode: reference image was None.")

                if self.params["reference_method"] == 2:
                    self.reference_image = None
                if self.params["add_drift_metrics"]:
                    if self.params["metric_nm_or_px"] == 0:
                        input_img.scalar_metrics["drift (nm)"] = (input_img.translation[0]**2 + input_img.translation[1]**2)**0.5 * input_img.pixel_size
                        input_img.scalar_metrics["x drift (nm)"] = input_img.translation[0] * input_img.pixel_size
                        input_img.scalar_metrics["y drift (nm)"] = input_img.translation[1] * input_img.pixel_size
                    else:
                        input_img.scalar_metrics["drift (px)"] = (input_img.translation[0] ** 2 + input_img.translation[1] ** 2) ** 0.5
                        input_img.scalar_metrics["x drift (px)"] = input_img.translation[0]
                        input_img.scalar_metrics["y drift (px)"] = input_img.translation[1]
                input_img.bake_transform(interpolation=1 if self.params["interpolation"]==0 else 3, edges=RegisterNode.edge_fill_options_scipy_argument[self.params["edge_fill"]], preserve_range=self.params["preserve_range"])
                return input_img
            elif self.params["register_method"] == 2:
                # set template if necessary
                if not self.pygpureg_template_image_set:
                    if self.params["reference_method"] == 0:
                        reference_image = self.connectable_attributes["image_in"].get_incoming_node().get_image(idx=None)
                        #self.pygpureg_template_image_set = True
                    elif self.params["reference_method"] == 1:
                        reference_image = data_source.get_image(self.params["frame"])
                        #self.pygpureg_template_image_set = True
                    else:
                        reference_image = data_source.get_image(idx - 1)

                    # make reference image the right size
                    max_size = min([RegisterNode._closest_power_of_2_below_n(reference_image.load().shape[0]), RegisterNode._closest_power_of_2_below_n(reference_image.load().shape[1])])
                    if max_size < self.params["pyGPUreg_size"]:
                        self.roi = [0, 0, max_size, max_size]
                        self.params["pyGPUreg_size"] = max_size
                    else:
                        self.roi = [0, 0, self.params["pyGPUreg_size"], self.params["pyGPUreg_size"]]

                    reference_image = reference_image.load_roi(self.roi)
                    #print(f'Setting pyGPUreg image size to {self.params["pyGPUreg_size"]}')
                    #pyGPUreg.set_image_size(self.params["pyGPUreg_size"])
                    #pyGPUreg.set_template(reference_image)
                # if the template is OK, the ROI is ok, so just grab the incoming image and register.

                pyGPUreg.register(np.random.uniform(0, 100, (256, 256)), np.random.uniform(0, 100, (256, 256)))
                #input_img_data = input_img.load_roi(self.roi)
                #input_img_data_registered, shift = pyGPUreg.register_to_template(input_img_data, edge_mode=self.params["edge_fill"])
                #input_img.data = input_img_data_registered
                #input_img.translation = [shift[0], shift[1]]
                #input_img.width = self.params["pyGPUreg_size"]
                # if self.params["add_drift_metrics"]:
                #     if self.params["metric_nm_or_px"] == 0:
                #         input_img.scalar_metrics["drift (nm)"] = (input_img.translation[0]**2 + input_img.translation[1]**2)**0.5 * input_img.pixel_size
                #         input_img.scalar_metrics["x drift (nm)"] = input_img.translation[0] * input_img.pixel_size
                #         input_img.scalar_metrics["y drift (nm)"] = input_img.translation[1] * input_img.pixel_size
                #     else:
                #         input_img.scalar_metrics["drift (px)"] = (input_img.translation[0] ** 2 + input_img.translation[1] ** 2) ** 0.5
                #         input_img.scalar_metrics["x drift (px)"] = input_img.translation[0]
                #         input_img.scalar_metrics["y drift (px)"] = input_img.translation[1]


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

        tmat, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC if method == 0 else cv2.LMEDS, confidence=confidence)
        return [tmat[0][2], tmat[1][2]]

    @staticmethod
    def _closest_power_of_2_below_n(N):
        logN = int(np.floor(np.log2(N)))
        return 2**logN

