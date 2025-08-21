#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
import time
import utils
import random
from tensorflow.keras import layers, models

random.seed(3)

# Model Configurations
image_size = (224, 224)
batch_size = 16
in_shape = (224, 224, 3)
lr = 0.001
epochs = 700
models_name = "mobilenet_v3_small"

# Data Paths
TRAINING_DIR = 'split_2_5_25_split_70/train'
VALIDATION_DIR = 'split_2_5_25_split_70/valid'
Test_DIR = 'split_2_5_25_split_70/test'

if __name__ == '__main__':
    # Data Loading and Preprocessing
    pre_in_model = tf.keras.applications.mobilenet_v3.preprocess_input
    train_data_gen, valid_data_gen, test_data_gen, class_num = utils.load_data(
        image_size, batch_size, TRAINING_DIR, VALIDATION_DIR, Test_DIR, pre_in_model
    )
    print(train_data_gen)

    # Model Build
    inputs = layers.Input(shape=in_shape)
    outputs = MobileNetV3Small(include_top=True, weights=None, classes=class_num)(inputs)
    model = tf.keras.Model(inputs, outputs)

    # Compile Model
    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Logging and Callbacks
    checkpoint, tb_callback = utils.logs(models_name)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=50,   # Stop if no improvement after 15 epochs
        restore_best_weights=True
    )

    # Train Model
    t0 = time.time()
    history = model.fit(
        train_data_gen, validation_data=valid_data_gen, epochs=epochs,
        verbose=1, callbacks=[tb_callback, checkpoint, early_stopping]
    )
    print("Training time:", time.time() - t0)

    # Plot Training History
    utils.plot_history(history, models_name)

    # Save Model
    filename = 'saved_model/custom_mobilenetv3.h5'
    model.save(filename)

    # Test Model
    utils.test_model(model, test_data_gen, models_name)
