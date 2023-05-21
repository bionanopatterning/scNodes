from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize

title = "Eman2"

def create(input_shape):
    inputs = Input(input_shape)

    # encoding
    conv1 = Conv2D(40, (15, 15), activation = 'relu', padding='same')(inputs)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(40, (15, 15), activation = 'relu', padding='same')(max1)
    upsample = Lambda(lambda x: resize(x, input_shape[:2]))(conv2)
    conv3 = Conv2D(1, (1, 1), activation = 'sigmoid', padding='same')(upsample)

    model = Model(inputs=[inputs], outputs=[conv3])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model


