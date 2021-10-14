import os
from PIL.Image import Image
from utils.xmlUtils import get_attribute_from_xml, get_bound_box_object
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications import inception_v3 as inc_net
from skimage import io
from skimage.transform import resize
import skimage


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
