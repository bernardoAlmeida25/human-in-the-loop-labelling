
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras import Sequential, metrics
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from skimage.segmentation import mark_boundaries
from tensorflow.keras import layers
from skimage import io
from skimage.transform import resize
from tensorflow.keras.preprocessing import image
import numpy as np
import pathlib
from lime import lime_image
import os
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from tensorflow_addons import metrics as tfa



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


def create_model(num_classes):
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss=BinaryCrossentropy(),
                  metrics=[metrics.Recall(), metrics.Precision(), metrics.AUC(), tfa.F1Score(num_classes=1)])

    return model


if __name__ == '__main__':
    # directory = show_select_folder_dialog()
    dest_dir = "/Users/bernardoalmeida/Documents/Dev/mask_detection"
    # crop_images_from_directory(directory, dest_dir)
    url = pathlib.Path("/Users/bernardoalmeida/Documents/Tese/to_label/663597.jpg")
    images = transform_img_fn([os.path.join('/Users/bernardoalmeida/Documents/Tese/to_label', '681375.jpg')])
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

    model = create_model(num_classes=1)

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs, verbose=1
    )

    saved_model = model.save_weights('testing/checkpoint')

    model.save_weights("ckpt.h5")
    v3 = tf.keras.applications.inception_v3.InceptionV3(include_top=True,
                                                        weights=model.load_weights("ckpt.h5"),
                                                        input_tensor=None,
                                                        classifier_activation='softmax',
                                                        classes=2)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(images[0].astype('double'), v3.predict, top_labels=2, hide_color=0,
                                             num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
    preds = v3.predict(images)
    prediction = np.argmax(preds)
    print(prediction)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title("Explanation for predicted class: " + str(explanation.top_labels[0]))
    plt.show()
