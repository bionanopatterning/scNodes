from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam

title = "VGGNet heavy"
include = False


def create(input_shape):
    inputs = Input(input_shape)

    # Encoder
    # Block 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 2
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Block 3
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    # Block 4
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)

    # Decoder
    # Block 5
    up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(pool4)
    merge1 = Concatenate([up1, conv8])
    conv9 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge1)

    # Block 6
    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv9)
    merge2 = Concatenate([up2, conv6])
    conv10 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge2)

    # Block 7
    up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv10)
    merge3 = Concatenate([up3, conv4])
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge3)

    # Block 8
    up4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv11)
    merge4 = Concatenate([up4, conv2])
    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge4)

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv12)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model