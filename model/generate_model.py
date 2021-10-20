from keras import Sequential, metrics
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow_addons import metrics as tfa
from tensorflow.keras import layers


def create_model(img_height, img_width, num_classes):
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss=BinaryCrossentropy(),
                  metrics=[metrics.Recall(), metrics.Precision(), metrics.AUC(), tfa.F1Score(num_classes=1)])

    return model
