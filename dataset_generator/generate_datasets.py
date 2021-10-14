from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory


def generate_train_dataset(directory, img_height, img_width, batch_size):
    return image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)


def generate_test_dataset(directory, img_height, img_width, batch_size):
    return image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

