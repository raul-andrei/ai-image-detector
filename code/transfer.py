import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = "data/train",
    labels = "inferred",
    batch_size = 32,
    image_size = (96, 96),
    interpolation='nearest'

)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = "data/test",
    labels = "inferred",
    batch_size = 32,
    image_size = (96, 96),
    interpolation='nearest'

)



base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./127.5, offset=-1, input_shape=(96, 96, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, epochs=5, validation_data=test_data)

model.evaluate(test_data, verbose=2)