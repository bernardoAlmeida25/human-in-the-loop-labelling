import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.segmentation import mark_boundaries
import pathlib
from lime import lime_image
import os
from dataset_generator.generate_datasets import generate_train_dataset, generate_test_dataset
from model.generate_model import create_model
from utils.filesystemUtils import show_select_folder_dialog
from utils.imageUtils import transform_img_fn, crop_images_from_directory

if __name__ == '__main__':
    directory = show_select_folder_dialog()
    dest_dir = "/Users/bernardoalmeida/Documents/Dev/mask_detection"
    crop_images_from_directory(directory, dest_dir)
    url = pathlib.Path("/Users/bernardoalmeida/Documents/Tese/to_label/663597.jpg")
    images = transform_img_fn([os.path.join('/Users/bernardoalmeida/Documents/Tese/to_label', '681375.jpg')])
    batch_size = 10
    img_height = 180
    img_width = 180

    train_ds = generate_train_dataset(directory=dest_dir,
                                      img_width=img_width,
                                      img_height=img_height,
                                      batch_size=batch_size)

    val_ds = generate_test_dataset(directory=dest_dir,
                                   img_width=img_width,
                                   img_height=img_height,
                                   batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    model = create_model(img_height=img_height, img_width=img_width, num_classes=1)

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs, verbose=1
    )

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

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title("Explanation for predicted class: " + str(explanation.top_labels[0]))
    plt.show()
