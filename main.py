import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from skimage.segmentation import mark_boundaries
import pathlib
from datetime import datetime
from PIL import ImageFont
import visualkeras
import pprint
import numpy as np
from lime import lime_image
import os
import time
from dataset_generator.generate_datasets import generate_train_dataset, generate_test_dataset
from model.generate_model import create_model, create_second_model, create_third_model, generate_plot
from utils.filesystemUtils import show_select_folder_dialog
from utils.imageUtils import transform_img_fn, crop_images_from_directory

if __name__ == '__main__':
    #directory = show_select_folder_dialog() DIALOG TO SELECT THE DATASET FOLDER
    dest_dir = os.path.join("PATH_TO_DATASET_FOLDER")
    #crop_images_from_directory(directory, dest_dir) CROP IMAGES TO SIZE IN THE XML
    images = transform_img_fn([os.path.join('PATH_TO_THE_IMAGE_TO_EXPLAIN', 'IMAGE.jpg')])
    batch_size = 10
    img_height = 180
    img_width = 180

    train_ds = generate_train_dataset(directory=dest_dir,
                                      img_width=img_width,
                                      img_height=img_height)

    val_ds = generate_test_dataset(directory=dest_dir,
                                   img_width=img_width,
                                   img_height=img_height)

    model1 = create_model(img_height=img_height, img_width=img_width, num_classes=2)

    model2 = create_second_model(img_height=img_height, img_width=img_width, num_classes=2)

    model3 = create_third_model(img_height=img_height, img_width=img_width, num_classes=2)

    callback = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

    epochs = 50
    t1 = datetime.utcnow()
    history = model1.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[callback])
    t2 = datetime.utcnow()
    delta = t2 - t1
    print(delta.total_seconds() / 60)

    def run_length(metric): return len(history.history[metric])

    number_of_epochs_it_ran = range(1, run_length('loss') + 1)

    last_loss_value = history.history['val_loss'][-1]
    last_precision_value = history.history['val_precision'][-1]
    last_recall_value = history.history['val_recall'][-1]
    last_auc_value = history.history['val_auc'][-1]
    last_f1_value = history.history['val_f1_score'][-1]

    print(last_loss_value, last_precision_value, last_recall_value, last_auc_value, last_f1_value)

    recall = history.history['recall']
    val_recall = history.history['val_recall']

    precision = history.history['precision']
    val_precision = history.history['val_precision']

    auc = history.history['auc']
    val_auc = history.history['val_auc']

    f1_score = history.history['f1_score']
    val_f1_score = history.history['val_f1_score']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    generate_plot(epochs=number_of_epochs_it_ran, metric=loss, val_metric=val_loss, val_name="Loss")
    generate_plot(epochs=number_of_epochs_it_ran, metric=recall, val_metric=val_recall, val_name="Recall")
    generate_plot(epochs=number_of_epochs_it_ran, metric=precision, val_metric=val_precision, val_name="Precision")
    generate_plot(epochs=number_of_epochs_it_ran, metric=auc, val_metric=val_auc, val_name="AUC")
    generate_plot(epochs=number_of_epochs_it_ran, metric=f1_score, val_metric=val_f1_score, val_name="F1 Score")


    model2.save_weights("ckpt.h5")
    v3 = tf.keras.applications.inception_v3.InceptionV3(include_top=True,
                                                         weights=model2.load_weights("ckpt.h5"),
                                                         input_tensor=None,
                                                         classifier_activation='softmax',
                                                         classes=2)
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(images[0].astype('double'), v3.predict, top_labels=2, hide_color=0,
                                          num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                 hide_rest=False)

    class_predicted = "mask" if explanation.top_labels[0] == 0 else "no mask"
    plt.axis("off")
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title("Explanation for predicted class: " + str(class_predicted))
    plt.show()
