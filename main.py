import glob
import os
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from xml.etree.ElementTree import ElementTree
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras import Sequential
from keras_preprocessing.image import ImageDataGenerator
from skimage.segmentation import mark_boundaries
from tensorflow.keras import layers
from excpetions.NoMatchingFilesException import NoMatchingFilesException
from dataset_generator import generate_datasets
from model import generate_model
from skimage import io
from skimage.transform import resize
from tensorflow.keras.preprocessing import image
import numpy as np
import pathlib
from lime import lime_image
from keras.preprocessing import image
import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt


# TODO: Refactor to generic module method
def show_select_folder_dialog():
    root = tk.Tk()
    root.withdraw()
    mb.showwarning(title='Select Directory', message='Select the same directory as previous selected in LabelImg')
    directory = fd.askdirectory()
    return directory


def validate_directory(directory, extension):
    path = directory + '/' + '*.' + extension
    if len(glob.glob(path)) == 0:
        raise NoMatchingFilesException(path=path, extension=extension)
    else:
        return path


def get_attribute_from_xml(path_to_file, path_to_attribute):
    tree = ElementTree(file=path_to_file)
    root = tree.getroot()
    object_xml = root.findall(path_to_attribute)
    return object_xml


def get_bound_box_object(path_to_file):
    bounded_box_object = get_attribute_from_xml(path_to_file, "object/bndbox/*")
    bounded_box_dict = {}
    for i in bounded_box_object:
        bounded_box_dict[i.tag] = i.text
    return bounded_box_dict


def crop_images_from_directory(directory, dest_dir):
    for file in os.listdir(directory):
        if file.endswith('.xml'):
            full_path = str(directory + '/' + file)
            image_path = get_attribute_from_xml(full_path, "path")
            label = get_attribute_from_xml(full_path, "object/name").pop().text
            image = image_path.pop().text
            image_directory = get_bound_box_object(full_path)
            filename = os.path.basename(str(directory + '/' + file))
            with Image.open(image) as img:
                (xmin, ymin, xmax, ymax) = (image_directory['xmin'],
                                            image_directory['ymin'],
                                            image_directory['xmax'],
                                            image_directory['ymax'])
                cropped_img = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                cropped_img.save(str(dest_dir + "/" + label + "/" + filename.split('.')[0] + '.jpg'))
        else:
            print(str(file + " not compatible"))


def create_dest_folders():
    dest_directory = os.path.join(os.path.dirname(os.getcwd()), "mask_detection")
    if os.path.exists(dest_directory):
        os.chdir(dest_directory)
        os.mkdir("mask")
        os.mkdir("no_mask")
    else:
        os.mkdir(dest_directory)
        os.chdir(dest_directory)
        os.mkdir("mask")
        os.mkdir("no_mask")
    return dest_directory

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

def read_and_transform_img(url):
    img = skimage.io.imread(url)
    img = skimage.transform.resize(img, (100, 100))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return img


if __name__ == '__main__':
    # directory = show_select_folder_dialog()
    dest_dir = "/Users/bernardoalmeida/Documents/Dev/mask_detection"
    # crop_images_from_directory(directory, dest_dir)
    url = pathlib.Path("/Users/bernardoalmeida/Documents/Tese/to_label/663597.jpg")
    images = transform_img_fn([os.path.join('/Users/bernardoalmeida/Documents/Tese/to_label', '663597.jpg')])
    batch_size = 10
    img_height = 180
    img_width = 180

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dest_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dest_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = 5

    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(20, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs, verbose=1
    )

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(images[0].astype('double'), model.predict,
                                             top_labels=2, hide_color=0, num_samples=1000)

    temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=True)
    temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                    hide_rest=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    ax1.imshow(mark_boundaries(temp_1, mask_1))
    ax2.imshow(mark_boundaries(temp_2, mask_2))
    ax1.axis('off')
    ax2.axis('off')