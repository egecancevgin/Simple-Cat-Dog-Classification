import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
import tensorflow as tf


def check_dataset(train, metadata):
    """ Checks the dataset and plots two example image."""

    get_label_name = metadata.features['label'].int2str

    for image, label in train.take(10):
        plt.figure()

        plt.imshow(image)

        plt.title(get_label_name(label))

    return 0


def format_example(image, label):
    """ Returns the image after reshaping the input image to the target image size."""

    img_size = 160

    image = tf.cast(image, tf.float32)

    image = (image / 127.5) - 1

    image = tf.image.resize(image, (img_size, img_size))

    return image, label


def check_shape(raw, train):
    """ Checks the image shapes to see if the reshaping worked in selected images."""

    for img, label in raw.take(5):
        print("Original shape:", img.shape)

    for img, label in train.take(5):
        print("New shape:", img.shape)

    return 0


def check_model(model, train_batch):
    """ Checks the model if it is working with the shape."""

    for img, _ in train_batch.take(1):
        pass

    return model(img).shape


