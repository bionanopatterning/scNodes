from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.optimizers import Adam

title = "ResNet"

def create(input_shape):
    inputs = Input(input_shape)

    # Residual block 1
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    shortcut = Conv2D(64, (1, 1), padding='same')(inputs)
    add1 = Add()([shortcut, conv1])
    act1 = Activation('relu')(add1)

    # Residual block 2
    conv2 = Conv2D(128, (3, 3), padding='same')(act1)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    shortcut2 = Conv2D(128, (1, 1), padding='same')(act1)
    add2 = Add()([shortcut2, conv2])
    act2 = Activation('relu')(add2)

    output = Conv2D(1, (1, 1), activation='sigmoid')(act2)

    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model