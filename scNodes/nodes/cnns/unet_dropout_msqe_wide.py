from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam

title = "UNet dropout wide (MSqE)"


def create(input_shape):
    inputs = Input(input_shape)

    # encoding
    conv1 = Conv2D(64, (7, 7), activation='relu', padding='same')(inputs)
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