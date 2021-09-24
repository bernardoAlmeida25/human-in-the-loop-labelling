from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

from dataset_generator import generate_datasets


def explainer():
    return lime_image.LimeImageExplainer()


def draw_heatmap(images, model):
    explanation = explainer().explain_instance(images[0].astype('double'),
                                               model.predict,
                                               top_labels=2,
                                               hide_color=0,
                                               num_samples=1000)
    temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=True)
    temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                    hide_rest=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    ax1.imshow(mark_boundaries(temp_1, mask_1))
    ax2.imshow(mark_boundaries(temp_2, mask_2))
    ax1.axis('off')
    ax2.axis('off')

def keras_training():

    url = pathlib.Path("/Users/bernardoalmeida/Documents/Tese/to_label/663597.jpg")
    images = read_and_transform_img(url)

    batch_size = 32
    img_height = 180
    img_width = 180

    train_dataset = generate_datasets.generate_train_dataset(directory=dest_dir, img_height=img_height, img_width=img_width,
                                                             batch_size=batch_size)
    validation_dataset = generate_datasets.generate_test_dataset(directory=dest_dir, img_height=img_height, img_width=img_width,
                                                                 batch_size=batch_size)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 200)
    normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    class_names = train_dataset.class_names

    model = generate_model.generate_model(img_height=img_height, img_width=img_width, num_classes=2)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(images[0].astype('double'), model.predict,
                                             top_labels=3, hide_color=0, num_samples=1000)
    temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=True)
    temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                    hide_rest=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    ax1.imshow(mark_boundaries(temp_1, mask_1))
    ax2.imshow(mark_boundaries(temp_2, mask_2))
    ax1.axis('off')
    ax2.axis('off')
