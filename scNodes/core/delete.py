from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam

title = "Denoising"

def create(input_shape):
    inputs = Input(input_shape)

    # encoding
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    drop1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    drop2 = Dropout(0.1)(conv2)

    # decoding
    up3 = Concatenate()([UpSampling2D(size=(2, 2))(drop2), drop1])
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)

    output = Conv2D(1, (1, 1), activation='linear')(conv3)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    return model

import tifffile
import matplotlib.pyplot as plt
import numpy as np

data = tifffile.imread("C:/Users/mart_/Desktop/nope.scnt")

N = data.shape[0]
S = data.shape[1]

model = create((S, S, 1))

epochs = 50
batchsize = 32

#
x = data[N//2:, :, :, None]
y = data[:N//2, :, :, None]
# print(x.shape)
# print(y.shape)
model.fit(x, y, epochs=epochs, batch_size=batchsize)

_y = model.predict(y)

for i in range(10):
    img_in = np.squeeze(y[i])
    img_out = np.squeeze(_y[i])
    plt.subplot(1, 2, 1)
    plt.imshow(img_in)
    plt.title("input")
    plt.subplot(1, 2, 2)
    plt.imshow(img_out)
    plt.title("output")
    plt.show()
# # apply multiple times
# x = np.reshape(data[1], (1, 64, 64, 1))
# y1 = model.predict(x)
# y2 = model.predict(y1)
# y3 = model.predict(y2)
# y4 = model.predict(y3)
# y5 = model.predict(y4)
#
# plt.subplot(1,6, 1)
# plt.imshow(np.squeeze(x))
# plt.subplot(1,6, 2)
# plt.imshow(np.squeeze(y1))
# plt.subplot(1,6, 3)
# plt.imshow(np.squeeze(y2))
# plt.subplot(1,6, 4)
# plt.imshow(np.squeeze(y3))
# plt.subplot(1,6, 5)
# plt.imshow(np.squeeze(y4))
# plt.subplot(1,6, 6)
# plt.imshow(np.squeeze(y5))
# plt.show()
