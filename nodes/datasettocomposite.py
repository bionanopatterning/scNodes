from node import *

def create():
    return DeinterleaveNode()

class DeinterleaveNode(Node):
    title = "Deinterleave"
    group = "Converters"
    size = 200
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)

    def __init__(self):
        super().__init__()

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)

        self.channels = list()

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
                self.channels.append(DeinterleaveOutputChannel(self))

            imgui.spacing()
            super().render_end()

    def get_image_impl(self, idx=None):
        print("Deinterleave get image")
        # Since this node outputs multiple datasets, we've got a new problem: reaching it via one of the dataset output's 'get_incoming_node' will return the node, but we won't know which dataset was addressed
        # So I added a thing: every connectable attribute now has a .latest_request = datetime obj that is set whenever get_incoming_node is called.
        active_channel = None
        _latest_request = self.channels[0].dataset_out.latest_request
        for channel in self.channels:
            if channel.dataset_out.latest_request >= _latest_request:
                active_channel = channel
                _latest_request = channel.dataset_out.latest_request
        print(active_channel.name)

        return active_channel.get_channel_img(idx)



class DeinterleaveOutputChannel:
    """Helper class of the DeinterleaveNode"""
    idgen = count(1)

    def __init__(self, parent):
        self.id = next(DeinterleaveOutputChannel.idgen)
        self.parent = parent
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=parent)

        self.index_str = "0"
        self.index = [0, 0]
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
        imgui.push_id(f"deinterleavechannel{self.id}")
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text(self.name)
        self.dataset_out.render_start()
        self.dataset_out.render_end()

        imgui.push_item_width(40)
        _c, self.index_str, = imgui.input_text("index", self.index_str, 256)
        _any_change = _c or _any_change
        if _c and len(self.index_str)>0:
            try:
                if "-" in self.index_str and self.index_str[-1]!='-':
                    vals = self.index_str.split("-")
                    self.index[0] = int(vals[0])
                    self.index[1] = int(vals[1])
                else:
                    self.index[0] = int(self.index_str)
                    self.index[1] = int(self.index_str)
            finally:
                pass
        Node.tooltip("The index of the frame of interest within the group. Enter either\n"
                     "a single number, or a range in the format: '1-10'. First frame of\n"
                     "a group has index 0.")

        content_width = imgui.get_window_content_region_width()
        imgui.same_line(position=content_width-10)
        if imgui.button("x", 20, 20):
            imgui.pop_id()
            self.delete()
            return True, True

        imgui.pop_id()
        return _any_change, False

    def delete(self):
        self.dataset_out.disconnect_all()
        self.dataset_out.delete()

    def get_channel_img(self, idx):
        datasource = self.parent.dataset_in.get_incoming_node()
        if datasource:
            n = self.parent.repeat_size
            # either return the indexed frame if that frame is in this channel
            if idx % n in range(self.index[0], self.index[1] + 1):
                return datasource.get_image(idx)
            else:
                # ugly way of doing it but this is easy:
                for i in range(0, n):
                    if i in range(self.index[0], self.index[1] + 1):
                        alt_idx = (idx // n) * n + i
                        return datasource.get_image(alt_idx)