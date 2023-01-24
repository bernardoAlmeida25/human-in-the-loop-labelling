from keras import Sequential, metrics
from tensorflow_addons import metrics as tfa
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt
from keras.layers import Rescaling
import numpy as np
from keras import backend as K


def create_model(img_height, img_width, num_classes):
    model = Sequential([
        Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[metrics.Recall(), metrics.Precision(), metrics.AUC(),
                           tfa.F1Score(num_classes=2, average='micro')])

    return model


def create_second_model(img_height, img_width, num_classes):
    model = Sequential([
        Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[metrics.Recall(), metrics.Precision(), metrics.AUC(),
                           tfa.F1Score(num_classes=2, average='micro')])

    return model


def create_third_model(img_height, img_width, num_classes):
    model = Sequential([
        Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[metrics.Recall(), metrics.Precision(), metrics.AUC(),
                           tfa.F1Score(num_classes=2, average='micro')])

    return model


def generate_plot(epochs, metric, val_metric, val_name):
    plt.figure(figsize=(7, 4))
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1.0))
    plt.plot(epochs, metric, label=f'Training {val_name}')
    plt.plot(epochs, val_metric, label=f'Validation {val_name}')
    plt.legend(loc='upper left')
    plt.xlabel("Epochs")
    plt.title(f'Training and Validation {val_name}')
    plt.show()
