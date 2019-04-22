from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, ReLU
from keras.models import Model

old_vgg = VGG16(weights="imagenet", include_top=False, input_shape=(240, 240, 3))
x = GlobalAveragePooling2D()(old_vgg.layers[-2].output)
x = Dense(64)(x)
x = BatchNormalization()(x)
x = ReLU()(x)

new_output = Dense(1, activation="sigmoid")(x)

vgg = Model(inputs=old_vgg.inputs, outputs=new_output, name="BeautyPredictor")
vgg.compile(optimizer="adam", loss="binary_crossentropy")
