from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize

title = "Mini"

def create(input_shape):
    inputs = Input(input_shape)

    # encoding
    conv1 = Conv2D(64, (5, 5), activation = 'relu', padding='same')(inputs)
    conv2 = Conv2D(64, (9, 9), strides=2, activation='relu', padding='same')(inputs)
    conv3 = Conv2D(64, (15, 15), strides=3, activation='relu', padding='same')(inputs)

    upsample2 = Lambda(lambda x: resize(x, input_shape[:2]))(conv2)
    upsample3 = Lambda(lambda x: resize(x, input_shape[:2]))(conv3)

    conc = Concatenate()([conv1, upsample2, upsample3])

    conv4 = Conv2D(1, (1, 1), activation='sigmoid')(conc)

    model = Model(inputs=[inputs], outputs=[conv4])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model


