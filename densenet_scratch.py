#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
import time
import utils
import random 
from tensorflow.keras import layers, models, Sequential
from tensorflow.python.keras.layers import Dense, Flatten


random.seed(3)


# In[6]:


image_size = (224, 224)
batch_size = 16
in_shape = (224, 224,3)
lr = 0.001
epochs=700
models_name="densenet_scratch"

# In[7]:


TRAINING_DIR = 'split_2_5_25_split_70/train'
VALIDATION_DIR = 'split_2_5_25_split_70/valid'
Test_DIR = 'split_2_5_25_split_70/test'


# In[8]:


if __name__ == '__main__':
    #Data loading and preprocesing
    pre_in_model = tf.keras.applications.densenet.preprocess_input
    train_data_gen, valid_data_gen,test_data_gen, class_num = utils.load_data(image_size, batch_size, 
                                                TRAINING_DIR, VALIDATION_DIR, Test_DIR, pre_in_model)
    print(train_data_gen)
    # Model build
    inputs=layers.Input(shape=in_shape)
    outputs=tf.keras.applications.DenseNet121(include_top=True,weights=None,classes=class_num)(inputs)
    model=tf.keras.Model(inputs,outputs)
    
    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])    
    model.summary()

    #model_name = densenet_model.name
    checkpoint, tb_callback=utils.logs(models_name)
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,   # Stop if no improvement after 50 epochs
    restore_best_weights=True
    )

    #balanced_train_data_gen = utils.apply_adasyn(train_data_gen, class_num, batch_size)

    # Model fitting
    t0 = time.time()
    history=model.fit(train_data_gen, validation_data=valid_data_gen, epochs=epochs, verbose=1, callbacks=[tb_callback, checkpoint, early_stopping])
    print("Training time:", time.time()-t0)

    utils.plot_history(history, models_name)
    
    # Saving model
    filename = 'saved_model/custom_resnet.h5'
    #model.save(filename)
    
    # Model testing
    utils.test_model(model, test_data_gen, models_name)
