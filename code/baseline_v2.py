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
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

model.fit(train_data, epochs=20, callbacks=[early_stop], validation_data=test_data)

model.evaluate(test_data, verbose=2)