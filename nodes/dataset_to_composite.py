from node import *

def create():
    return DeinterleaveNode()

class DeinterleaveNode(Node):
    title = "Deinterleave & make composite"
    group = "Miscellaneous"
    size = 200
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)

    def __init__(self):
        super().__init__()

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)

        self.channels = [DeinterleaveOutputChannel()]

        self.repeat_size = 10

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_in.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(40)
            _c, self.repeat_size = imgui.input_int("Repeat size", self.repeat_size, 0, 0)
            self.any_change = self.any_change or _c

            # render channels
            for channel in self.channels:
                _c, _delete = channel.render()
                self.any_change = self.any_change or _c
                if _delete:
                    self.channels.remove(channel)

            imgui.new_line()
            content_width = imgui.get_window_content_region_width()
            imgui.same_line(position=(content_width - 40) / 2)
            if imgui.button("+", width=40, height=20):
                self.channels.append(DeinterleaveOutputChannel())

            imgui.spacing()
            super().render_end()

    def get_image_impl(self, idx=None):
        datasource = self.dataset_in.get_incoming_node()
        if datasource:
            outframe = datasource.get_image(idx).clone()
            input_img = outframe.load()
            w = np.shape(input_img)[0]
            h = np.shape(input_img)[1]
            out_img = np.zeros((w, h, 3))
            for channel in self.channels:
                cidx = channel.index
                clr = channel.colour
                if idx % self.repeat_size in range(channel.index[0], channel.index[1]+1):
                    pxd = input_img
                else:
                    for i in range(0, self.repeat_size):
                        if i in range(channel.index[0], channel.index[1]+1):
                            pxd = datasource.get_image((idx // self.repeat_size) * self.repeat_size + i).load()
                for i in range(0, 3):
                    if clr[i] != 0:
                        out_img[:, :, i] += pxd * clr[i]
            outframe.data = out_img
            return outframe





class DeinterleaveOutputChannel:
    """Helper class of the DeinterleaveNode"""
    idgen = count(1)

    def __init__(self):
        self.id = next(DeinterleaveOutputChannel.idgen)

        self.index_str = "0"
        self.index = [0, 0]
        self.colour = (1.0, 1.0, 1.0)
        self.name = f"Channel {self.id}"

    def __eq__(self, other):
        if isinstance(other, DeinterleaveOutputChannel):
            return self.id == other.id
        return False

    def render(self):
        """
        :return: (changed, delete) tuple of bools.
        """
        _any_change = False
        content_width = imgui.get_window_content_region_width()
        imgui.push_id(f"deinterleavechannel{self.id}")
        imgui.spacing()

        imgui.text(self.name)

        imgui.same_line(position=content_width - 10)
        _c, self.colour = imgui.color_edit3("##Colour_min", *self.colour, imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL)
        _any_change = _c or _any_change
        imgui.push_item_width(40)
        _c, self.index_str, = imgui.input_text("index", self.index_str, 256)
        _any_change = _c or _any_change
        if _c and len(self.index_str)>0:
            try:
                if "-" in self.index_str and self.index_str[-1]!='-':
                    print("DASH")
                    print(self.index_str)
                    vals = self.index_str.split("-")
                    self.index[0] = int(vals[0].replace("-",""))
                    self.index[1] = int(vals[1])
                else:
                    print("NO DASH")
                    self.index[0] = int(self.index_str)
                    self.index[1] = int(self.index_str)
            except Exception as e:
                pass
        Node.tooltip("The index of the frame of interest within the group. Enter either\n"
                     "a single number, or a range in the format: '1-10'. First frame of\n"
                     "a group has index 0.")


        imgui.same_line(position=content_width-10)
        if imgui.button("x", 20, 20):
            imgui.pop_id()
            return True, True

        imgui.pop_id()
        return _any_change, False
