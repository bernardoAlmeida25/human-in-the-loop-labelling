from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def generate_model():
    return Sequential([

    # First convolution
        Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D(2, 2),

    # Second convolution
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

    # Third convolution
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),

    # Dense hidden layer
        Dense(512, activation='relu'),
        Dropout(0.2),

    # Output neuron.
        Dense(3, activation='softmax')
    ])


def compile_model(model: Sequential):
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

