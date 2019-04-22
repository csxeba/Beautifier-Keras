from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Activation, BatchNormalization, GlobalAveragePooling2D, concatenate


def dense_block(x0, width, depth):

    outputs = [x0]

    for d in range(depth):
        x = Conv2D(width, 3, padding="same")(outputs[-1])
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        outputs.append(x)

    return concatenate(outputs)


class FCN:

    def __init__(self):
        self.training_aspect = None  # type: Model
        self.testing_aspect = None  # type: Model

    def build_beaty_predictor(self, input_shape=(240, 240, 3)):
        inputs = Input(input_shape)

        x = Conv2D(16, 5, strides=(2, 2), padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = dense_block(x, width=16, depth=2)
        x = MaxPool2D()(x)  # 60

        x = dense_block(x, width=32, depth=3)
        x = MaxPool2D()(x)  # 30

        x = dense_block(x, width=64, depth=3)
        x = MaxPool2D()(x)  # 15

        out3 = Conv2D(1, 1)(x)
        out3 = GlobalAveragePooling2D()(out3)

        self.training_aspect = Model(inputs=inputs, outputs=out3, name="Beholder-Training")
        self.testing_aspect = Model(inputs=inputs, outputs=out3, name="Beholder-Testing")

        self.training_aspect.compile("adam", "mse")
