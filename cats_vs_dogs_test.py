from cats_vs_dogs_functions import *


def main():
    """ Driver function."""
    # Download the Cats and Dogs dataset from TensorFlow
    (raw_train, raw_val, raw_test), metadata = tfds.load(
        'cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True, as_supervised=True
    )
    # Metadata will contain descriptions, formats, author, etc.

    check_dataset(raw_train, metadata)

    IMG_SIZE = 160
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    train = raw_train.map(format_example)

    validation = raw_val.map(format_example)

    test = raw_test.map(format_example)

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    validation_batches = validation.batch(BATCH_SIZE)

    test_batches = test.batch(BATCH_SIZE)

    check_shape(raw_train, train)

    # Using a pre-trained model to create the base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    test_shape = check_model(base_model, train_batches)
    print("Checking the shape:", test_shape)

    # The base model layers will not be trainable anymore so that the Dense Layer can learn
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['Accuracy']
    )

    # The model can be evaluated right now to see how it does before training
    loss0, accuracy0 = model.evaluate(validation_batches, steps=20)

    # Training can be applied on the images
    history = model.fit(
        train_batches,
        epochs=3,
        validation_data=validation_batches
    )

    model.save('dogs_vs_cats.h5')

    new_model = tf.keras.models.load_model('dogs_vs_cats.h5')


if __name__ == '__main__':
    main()

