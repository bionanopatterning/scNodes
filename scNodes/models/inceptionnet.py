from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.optimizers import Adam

title = "InceptionNet"

def create(input_shape):
    inputs = Input(input_shape)

    # Apply transforms in parallel and concatenate results
    t1 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)

    t2 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
    t2 = Conv2D(64, (3, 3), padding='same', activation='relu')(t2)

    t3 = Conv2D(64, (1, 1), padding='same', activation='relu')(inputs)
    t3 = Conv2D(64, (5, 5), padding='same', activation='relu')(t3)

    t4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    t4 = Conv2D(64, (1, 1), padding='same', activation='relu')(t4)

    concatenated = Concatenate()([t1, t2, t3, t4])
    output = Conv2D(1, (1, 1), padding='same', activation="sigmoid")(concatenated)
    # create the model
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    return model