#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
import time
import utils
import random 
from tensorflow.keras import layers, models, Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras import layers, Model

# In[19]:

n=224
image_size=(n, n)
batch_size=16
epochs=700
models_name="hybridvtdense"

# In[20]:


TRAINING_DIR = 'split_2_5_25_split_70/train'
VALIDATION_DIR = 'split_2_5_25_split_70/valid'
Test_DIR = 'split_2_5_25_split_70/test'


pre_in_densenet = tf.keras.applications.densenet.preprocess_input
train_data_gen, valid_data_gen,test_data_gen, class_num = utils.load_data(image_size, batch_size, 
                                                TRAINING_DIR, VALIDATION_DIR, Test_DIR, pre_in_densenet)


# Define the hybrid model with a reduced DenseNet backbone
def create_hybrid_model(input_shape=(224, 224, 3), num_classes=3):
    """ Hybrid Model with Fewer DenseNet Layers + Transformer Block """
    
    # Load DenseNet but stop at an earlier layer (shallower model)
    densenet = DenseNet121(
        include_top=False,
        input_shape=input_shape
    )
    # Print all layer names to find the correct one
    for i, layer in enumerate(densenet.layers):
    	print(i, layer.name)
    # Reduce the number of layers (stop at an earlier stage)
    densenet = Model(inputs=densenet.input, outputs=densenet.get_layer("conv2_block6_concat").output)

    # Input Layer
    inputs = layers.Input(shape=input_shape)
    x = densenet(inputs)  # Shape: (batch_size, 14, 14, 896) instead of full 1024

    # Global Pooling
    x = layers.GlobalAveragePooling2D()(x)  # Shape: (batch_size, 896)

    # Ensure reshaping is compatible
    num_patches = 16
    projection_dim = x.shape[-1] // num_patches
    if x.shape[-1] % num_patches != 0:
        raise ValueError(f"Cannot reshape tensor of shape {x.shape[-1]} into {num_patches} patches.")
    x = layers.Reshape((num_patches, projection_dim))(x)

    # Apply Transformer Block (1 Layer)
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=8, key_dim=projection_dim)(x, x)

    # Global Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Classification Head
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

# Instantiate and compile the model
model = create_hybrid_model()
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model_name = model.name
checkpoint, tb_callback=utils.logs(models_name)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,   # Stop if no improvement after 50 epochs
    restore_best_weights=True
    )


t0 = time.time()
history = model.fit(
    train_data_gen,  # Your training data generator
    validation_data=valid_data_gen,  # Your validation data generator
    epochs=epochs,
    callbacks=[tb_callback, checkpoint, early_stopping],  # Add callbacks (e.g., ModelCheckpoint, EarlyStopping)
    verbose=1
)
print("Training time:", time.time()-t0)

# In[ ]:

utils.test_model(model, test_data_gen, models_name)
utils.plot_history(history , models_name)

