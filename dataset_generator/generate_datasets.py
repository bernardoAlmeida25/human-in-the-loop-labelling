from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   validation_split=0.1)


def generate_train_dataset(directory, img_height, img_width, batch_size):
    return image_dataset_from_directory(
        directory,
        validation_split=0.2,
        label_mode='categorical',
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)


def generate_test_dataset(directory, img_height, img_width, batch_size):
    return image_dataset_from_directory(
        directory,
        validation_split=0.2,
        label_mode='categorical',
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
