import glfw
from opengl_classes import *
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
from dataset import *
from util import *


class Reconstructor:
    default_pixel_size = 6.4
    default_uncertainty = 100.0 / 6.4 * default_pixel_size
    kernel_size = 128
    default_img_size = (5000, 5000)
    default_tile_size = (2048, 2048)
    vertices = [-0.5, 0.5, 0.0, 1.0, -0.5, -0.5, 0.0, 0.0, 0.5, -0.5, 1.0, 0.0, 0.5, 0.5, 1.0, 1.0]
    indices = [0, 1, 2, 2, 0, 3]

    def __init__(self):
        # Shader
        self.shader = Shader("shaders/reconstructor_shader.glsl")
        # create texture, gaussian spot img, set pixel sizes etc.
        self.texture = Texture(format="r32f")
        self.texture.set_linear_interpolation()

        self.pixel_size = Reconstructor.default_pixel_size  # nm per pixel
        self.quad_uncertainty = Reconstructor.default_uncertainty
        self.quad_size = Reconstructor.kernel_size
        self.image_size = Reconstructor.default_img_size
        self.tile_size = Reconstructor.default_tile_size
        self.set_pixel_size(Reconstructor.default_pixel_size)
        self.fbo = FrameBuffer(*self.tile_size, texture_format="rgb32f")
        self.vao = VertexArray(VertexBuffer(Reconstructor.vertices), IndexBuffer(Reconstructor.indices),
                               attribute_format="xyuv")
        self.instance_vbos = dict()
        self.particle_data = ParticleData()
        self.camera = StaticCamera(Reconstructor.default_tile_size)

        self.mode = "ui16"
        self.camera_origin = [0.0, 0.0]

    def set_mode(self, mode="float"):
        """
        :param mode: either "float" or "ui16." Save as 32 bit float for best quality, or as 16 bit unsigned int when
        using the render result in ImageViewer (rgb32f reconstructions quickly exceed GPU memory).
        """
        if self.mode in ["float", "ui16"]:
            self.mode = mode

    def set_pixel_size(self, pixel_size):
        """
        :param pixel_size: float, pixel size in nm for the final reconstruction.
        """
        self.pixel_size = pixel_size

        # Reset the default-gaussian-quad to use the new pixel size.
        default_sigma = self.default_uncertainty / self.pixel_size
        _g_img_axes = np.linspace(-(Reconstructor.kernel_size - 1) / 2., (Reconstructor.kernel_size - 1) / 2.,
                                  Reconstructor.kernel_size)
        _g_img_gauss = np.exp(-0.5 * np.square(_g_img_axes) / np.square(default_sigma))
        _g_img_kernel = np.outer(_g_img_gauss, _g_img_gauss)
        default_gaussian = _g_img_kernel / np.sum(_g_img_kernel)
        self.texture.update(default_gaussian)

    def set_image_size(self, image_size):
        self.image_size = [image_size[0] // 2 * 2, image_size[1] // 2 * 2]

    def set_particle_data(self, particle_data):
        """
        :param particle_data: object of type ParticleData. Sets the particle dataset for the reconstructor to render.
        :return:
        """
        self.particle_data = particle_data

    def update_particle_data(self):
        self.create_instance_buffers()
        self.particle_data.baked_by_renderer = True

    def recompile_shader(self):
        """
        Useful during debugging of the shaders. To be deleted in release.
        :return:
        """
        try:
            shader = Shader("shaders/reconstructor_shader.glsl")
            self.shader = shader
        except Exception as e:
            raise e

    def render(self, fixed_uncertainty=None):
        """
        Render the particles in the currently st ParticleData obj. (see set_particle_data()), at the current pixel size (see set_pixel_size()).
        :return: a 3D [width, height, 3] numpy array of type float (when in mode 'float') or uint16 (mode 'ui16'). See set_mode()
        """
        if not self.particle_data.baked_by_renderer:
            self.update_particle_data()  # refresh the instance buffers
        if fixed_uncertainty is not None:
            self.set_fixed_uncertainty_value(fixed_uncertainty)

        self.update_particle_colours()
        self.update_particle_states()
        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD)
        glBlendFunc(GL_ONE, GL_ONE)
        self.fbo.clear((0.0, 0.0, 0.0, 1.0))
        self.fbo.bind()
        self.vao.bind()
        self.shader.bind()
        self.shader.uniform1f("quad_pixel_size", Reconstructor.kernel_size)
        self.shader.uniform1f("quad_uncertainty", self.quad_uncertainty)
        self.shader.uniform1f("pixel_size", self.pixel_size)
        self.texture.bind(0)
        # Make empty image
        W = self.image_size[0]
        w = self.tile_size[0]
        H = self.image_size[1]
        h = self.tile_size[1]

        sr_image = np.zeros((H, W, 3), dtype=np.float32)
        print("sr image shape", sr_image.shape)
        tiles_x = int(np.ceil(W / w))
        tiles_y = int(np.ceil(H / h))
        for i in range(tiles_x):
            for j in range(tiles_y):
                print(i, j)
                tile = self.render_tile((i, j))
                sr_image[j * h:min([(j + 1) * h, H]), i * w:min([(i + 1) * w, W]), :] = tile[:min([h, H - j * h]), :min([w, W - i * w]), :]

        self.shader.unbind()
        self.vao.unbind()
        self.fbo.unbind()
        if fixed_uncertainty is not None:
            self.undo_fixed_uncertainty_value()
        if self.mode == "float":
            return sr_image
        elif self.mode == "ui16":
            sr_image_ui16 = np.zeros((H, W, 3), dtype = np.uint16)
            r = sr_image[:, :, 0] / (1 + np.amax(sr_image[:, :, 0])) * 65535
            g = sr_image[:, :, 1] / (1 + np.amax(sr_image[:, :, 1])) * 65535
            b = sr_image[:, :, 2] / (1 + np.amax(sr_image[:, :, 2])) * 65535
            sr_image_ui16[:, :, 0] = r.astype(np.uint16)
            sr_image_ui16[:, :, 1] = g.astype(np.uint16)
            sr_image_ui16[:, :, 2] = b.astype(np.uint16)
            return sr_image_ui16

    def set_camera_origin(self, camera_origin):
        """
        :param camera_origin: origin position in world coordinates for the camera.
        :return:
        """
        self.camera_origin = camera_origin
        print("CAMERA ORIGIN", self.camera_origin)

    def create_instance_buffers(self):
        self.vao.bind()

        self.instance_vbos["x"] = VertexBuffer(self.particle_data.parameter['x (nm)'])
        self.instance_vbos["x"].set_location_and_stride(2, 1)
        self.instance_vbos["x"].set_divisor_to_per_instance()

        self.instance_vbos["y"] = VertexBuffer(self.particle_data.parameter['y (nm)'])
        self.instance_vbos["y"].set_location_and_stride(3, 1)
        self.instance_vbos["y"].set_divisor_to_per_instance()

        self.instance_vbos["uncertainty"] = VertexBuffer(self.particle_data.parameter['uncertainty (nm)'])
        self.instance_vbos["uncertainty"].set_location_and_stride(4, 1)
        self.instance_vbos["uncertainty"].set_divisor_to_per_instance()

        self.instance_vbos["colour"] = VertexBuffer()
        self.instance_vbos["colour"].set_location_and_stride(5, 3)
        self.instance_vbos["colour"].set_divisor_to_per_instance()

        self.instance_vbos["state"] = VertexBuffer()
        self.instance_vbos["state"].set_location_and_stride(6, 1)
        self.instance_vbos["state"].set_divisor_to_per_instance()
        self.vao.unbind()

        self.update_particle_colours()

    def set_fixed_uncertainty_value(self, value):
        self.vao.bind()
        self.instance_vbos["uncertainty"].update(np.ones((self.particle_data.n_particles, 1)) * value)
        self.vao.unbind()

    def undo_fixed_uncertainty_value(self):
        self.vao.bind()
        self.instance_vbos["uncertainty"].update(self.particle_data.parameter['uncertainty'])
        self.vao.unbind()

    def update_particle_colours(self):
        self.vao.bind()
        _colours = list()
        for particle in self.particle_data.particles:
            _colours.append(particle.colour)
        self.instance_vbos["colour"].update(np.asarray(_colours).flatten(order = "C"))
        self.vao.unbind()

    def update_particle_states(self):
        self.vao.bind()
        _states = list()
        for particle in self.particle_data.particles:
            _states.append(particle.visible * 1.0)
        self.instance_vbos["state"].update(np.asarray(_states).flatten(order = "C"))
        self.vao.unbind()

    def render_tile(self, tile_idx=(0, 0)):
        self.fbo.clear((0.0, 0.0, 0.0, 1.0))
        self.fbo.bind()
        camera_x = -(tile_idx[0] + 0.5) * self.tile_size[0]
        camera_y = -(tile_idx[1] + 0.5) * self.tile_size[1]

        print("Tile size is ", self.tile_size)
        self.camera.position = [camera_x + self.camera_origin[0], camera_y + self.camera_origin[1], 0.0]
        print("Camera position x, y is: ", self.camera.position[0], self.camera.position[1])
        self.camera.update_matrix()
        self.shader.uniformmat4("cameraMatrix", self.camera.view_projection_matrix)
        glDrawElementsInstanced(GL_TRIANGLES, self.vao.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None,
                                self.particle_data.n_particles)
        tile = glReadPixels(0, 0, self.default_tile_size[0], self.default_tile_size[1], GL_RGB, GL_FLOAT)
        return tile



class StaticCamera:
    def __init__(self, sensor_size):
        self.position = np.asarray([0.0, 0.0, 0.0])
        self.zoom = 1.0
        self.view_matrix = np.identity(4)
        self.projection_matrix = np.identity(4)
        self.view_projection_matrix = np.identity(4)
        self.sensor_size = sensor_size
        self.update_matrix()

    def update_matrix(self):
        self.projection_matrix = np.matrix([
            [2 / self.sensor_size[0], 0, 0, 0],
            [0, 2 / self.sensor_size[1], 0, 0],
            [0, 0, -2 / 100, 0],
            [0, 0, 0, 1],
        ])
        self.view_matrix = np.matrix([
            [self.zoom, 0.0, 0.0, self.position[0]],
            [0.0, self.zoom, 0.0, self.position[1]],
            [0.0, 0.0, self.zoom, self.position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.view_projection_matrix = np.matmul(self.projection_matrix, self.view_matrix)
