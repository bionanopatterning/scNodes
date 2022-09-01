from opengl_classes import *


class ROI:
    def __init__(self, box=[0, 0, 1, 1], colour=(1.0, 1.0, 1.0, 1.0)):
        """

        :param box: list/tuple with (x_min, y_min, x_max, y_max) coordinates.
        :param colour: colour of the ROI
        """
        self.box = list(box)
        self.colour = colour
        self.va = VertexArray(None, None, attribute_format="xy")
        self.visible = True
        self.use = False
        self.update_va()

    def update_va(self):
        left = self.box[0]
        top = self.box[1]
        right = self.box[2]
        bottom = self.box[3]
        coordinates = [left, bottom,
                       right, bottom,
                       right, top,
                       left, top]
        indices = [0, 1, 1, 2, 2, 3, 3, 0]
        self.visible = not (left == right and top == bottom)
        self.va.update(VertexBuffer(coordinates), IndexBuffer(indices))

    def render(self, shader, camera):
        if self.visible:
            self.va.bind()
            shader.bind()
            shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
            shader.uniform3f("lineColour", self.colour)
            shader.uniform3f("translation", [0.0, 0.0, 0.0])
            glDrawElements(GL_LINES, self.va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            shader.unbind()
            self.va.unbind()

    def is_in_roi(self, point):
        return self.box[0] < point[0] < self.box[2] and self.box[1] < point[1] < self.box[3]

    def translate(self, shift):
        self.box[0] += int(shift[0])
        self.box[1] += int(shift[1])
        self.box[2] += int(shift[0])
        self.box[3] += int(shift[1])
        self.update_va()

    def set_box(self, box):
        self.box = box
        self.visible = True
        if self.box[0] == self.box[2] and self.box[1] == self.box[3]:
            self.visible = False
        self.update_va()

    def correct_order(self):
        change = False
        if self.box[2] < self.box[0]:
            self.box[0], self.box[2] = self.box[2], self.box[0]
            change = True
        if self.box[3] < self.box[1]:
            self.box[1], self.box[3] = self.box[3], self.box[1]
            change = True
        if change:
            self.update_va()

    def limit(self, width, height):
        self.box[0] = min([width, max([0, self.box[0]])])
        self.box[1] = min([height, max([0, self.box[1]])])
        self.box[2] = min([width, max([0, self.box[2]])])
        self.box[3] = min([height, max([0, self.box[3]])])
        self.update_va()
