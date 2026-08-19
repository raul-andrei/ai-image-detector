import tensorflow as tf

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = "data/train",
    labels = "inferred",
    batch_size = 32,
    image_size = (32, 32)

)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = "data/test",
    labels = "inferred",
    batch_size = 32,
    image_size = (32, 32)

)

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape = (32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, epochs=5, validation_data=test_data)

model.evaluate(test_data, verbose=2)