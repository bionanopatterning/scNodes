from __future__ import print_function, division
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, UpSampling2D, Conv2D, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import numpy as np

title = "Pix2pix"


def create(input_shape):
    return Pix2Pix(input_shape)


class Pix2Pix():
    def __init__(self, input_shape):
        # Input shape
        self.box_size = input_shape[0]
        self.img_shape = input_shape

        # Calculate output shape of D (PatchGAN)
        patch = int(self.box_size / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 16
        self.df = 16

        optimizer = Adam(2e-4, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate(axis=-1)([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)  # (64, 64, 1)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)  # (32, 32, 16)
        d2 = conv2d(d1, self.gf*2)  # (16, 16, 32)
        d3 = conv2d(d2, self.gf*4)  # (8, 8, 64)
        d4 = conv2d(d3, self.gf*8)  # (4, 4, 128)
        d5 = conv2d(d4, self.gf*8)  # (2, 2, 128)
        d6 = conv2d(d5, self.gf*8)  # (1, 1, 128)

        # Upsampling
        u1 = deconv2d(d6, d5, self.gf*8)
        u2 = deconv2d(u1, d4, self.gf*8)
        u3 = deconv2d(u2, d3, self.gf*4)
        u4 = deconv2d(u3, d2, self.gf*2)
        u5 = deconv2d(u4, d1, self.gf)

        u6 = UpSampling2D(size=2)(u5)
        output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(u6)
        model = Model(d0, output_img)
        return model


    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])  # (64, 64, 2)

        d1 = d_layer(combined_imgs, self.df, bn=False)  # (32, 32, 16)
        d2 = d_layer(d1, self.df*2)  # (16, 16, 32)
        d3 = d_layer(d2, self.df*4)  # (8, 8, 64)
        d4 = d_layer(d3, self.df*8)  # (4, 4, 128)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        model = Model(inputs=[img_A, img_B], outputs=validity)
        return model

    def fit(self, train_x, train_y, epochs, batch_size=1, shuffle=True, callbacks=[]):
        for c in callbacks:
            c.params['epochs'] = epochs
        n_samples = train_x.shape[0]
        for epoch in range(epochs):
            for c in callbacks:
                c.on_epoch_begin(epoch)
            indices = list(range(n_samples))
            if shuffle:
                np.random.shuffle(indices)
            for batch_i in range(n_samples // batch_size + 1):
                batch_indices = list()
                for i in range(min(batch_size, len(indices))):
                    batch_indices.append(indices.pop(-1))

                # Adversarial loss ground truths
                valid = np.ones((len(batch_indices), ) + self.disc_patch)
                fake = np.zeros((len(batch_indices), ) + self.disc_patch)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                imgs_X = train_x[batch_indices, :, :, :]
                imgs_Y = train_y[batch_indices, :, :, :]
                fake_Y = self.generator.predict(imgs_X)
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_X, imgs_Y], valid)
                d_loss_fake = self.discriminator.train_on_batch([imgs_X, fake_Y], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_X, imgs_Y], [valid, imgs_X])

                logs = dict()
                logs['loss'] = g_loss[0]
                for c in callbacks:
                    c.on_batch_end(batch_i, logs)


    def predict(self, images):
        return self.generator.predict(images)

    def count_params(self):
        return self.generator.count_params()


print ("[Epoch %d/%d] [Batch %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))