#!/usr/bin/env python
# coding: utf-8

# In[70]:


#import keras.preprocessing.image
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import os
import time
#from imblearn.over_sampling import ADASYN
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


# In[71]:

from PIL import UnidentifiedImageError, Image
import warnings

'''def load_data(image_size, batch_size, train_dir, valid_dir, test_dir, preprocess_input):
    training_datagen = SafeImageDataGenerator(
        rescale=1. / 255, shear_range=0.2,
        height_shift_range=0.1,
        vertical_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        #brightness_range=(0.5, 1.5),
        preprocessing_function=preprocess_input)

    train_dataset = training_datagen.flow_from_directory(train_dir, target_size=image_size,
                                                         batch_size=batch_size, class_mode='categorical')
    class_number = train_dataset.__next__()[1].shape[1] 
    
    validation_datagen = SafeImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input)
    validation_dataset = validation_datagen.flow_from_directory(valid_dir, target_size=image_size,
                                                                    batch_size=batch_size, class_mode='categorical')

    test_dataset = SafeImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input)
    test_generator = test_dataset.flow_from_directory(test_dir, target_size=image_size,
                                                          batch_size=1, class_mode='categorical', shuffle=False)

    return train_dataset, validation_dataset, test_generator, class_number'''

def load_data(image_size, batch_size, train_dir, valid_dir, test_dir, preprocess_input):
    training_datagen = ImageDataGenerator(
        rescale=1. / 255, shear_range=0.2,
        height_shift_range=0.1,
        vertical_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        #brightness_range=(0.5, 1.5),
        preprocessing_function=preprocess_input)

    train_dataset = training_datagen.flow_from_directory(train_dir, target_size=image_size,
                                                         batch_size=batch_size, class_mode='categorical')
    class_number = train_dataset.num_classes
    
    validation_datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input)
    validation_dataset = validation_datagen.flow_from_directory(valid_dir, target_size=image_size,
                                                                    batch_size=batch_size, class_mode='categorical')

    test_dataset = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input)
    test_generator = test_dataset.flow_from_directory(test_dir, target_size=image_size,
                                                          batch_size=1, class_mode='categorical', shuffle=False)

    return train_dataset, validation_dataset, test_generator, class_number




def apply_adasyn(train_data_gen, class_num, batch_size):
    """
    Applies Adaptive Synthetic Sampling (ADASYN) to balance the training dataset.

    Args:
        train_data_gen (ImageDataGenerator): Original training data generator.
        class_num (int): Number of classes in the dataset.
        batch_size (int): Batch size for the generator.

    Returns:
        ImageDataGenerator: A new generator with the balanced dataset.
    """
    # Step 1: Extract images and labels from the generator
    x_train, y_train = [], []

    for _ in range(train_data_gen.samples // batch_size + 1):
        images, labels = next(train_data_gen)
        x_train.extend(images)
        y_train.extend(labels)

    x_train = np.array(x_train)
    y_train = np.argmax(np.array(y_train), axis=1)  # Convert one-hot labels to class indices

    # Step 2: Flatten image data for ADASYN
    x_train_flat = x_train.reshape(x_train.shape[0], -1)

    # Step 3: Apply ADASYN
    adasyn = ADASYN(sampling_strategy='minority', random_state=42)
    x_train_resampled, y_train_resampled = adasyn.fit_resample(x_train_flat, y_train)

    # Step 4: Reshape the image data back to its original dimensions
    x_train_resampled = x_train_resampled.reshape(-1, *x_train.shape[1:])
    y_train_resampled = np.eye(class_num)[y_train_resampled]  # Convert back to one-hot encoding

    # Step 5: Create a new ImageDataGenerator with balanced data
    balanced_train_gen = ImageDataGenerator()
    balanced_train_data_gen = balanced_train_gen.flow(
        x_train_resampled,
        y_train_resampled,
        batch_size=batch_size
    )
    unique, counts = np.unique(np.argmax(balanced_train_data_gen.y, axis=1), return_counts=True)
    print("Balanced class distribution after ADASYN:")
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples")

    # Return the new generator
    return balanced_train_data_gen


# In[75]:

#def logs(model_name):
#    model_filepath=os.path.join("checkpoint", model_name)
#    checkpoint = ModelCheckpoint(os.path.join(model_filepath, "cp", f"{model_name}" + 
#                                              "-loss-{val_loss:.2f}.keras"), save_best_only=True, verbose=1)
#    tb_callback=TensorBoard(log_dir=os.path.join(model_filepath, "tb", model_name))
#    return checkpoint, tb_callback

def logs(model_name):
    # Create a unique timestamp for the model run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Define directory paths with model_name as a subfolder
    checkpoint_dir = os.path.join("checkpoint", model_name)  # e.g., checkpoint/densenet/
    tb_log_dir = os.path.join(checkpoint_dir, "tb")  # TensorBoard logs inside model folder

    # Ensure directories exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Define unique model file path inside the model's subfolder
    model_filepath = os.path.join(checkpoint_dir, f"{model_name}-{timestamp}-loss-{{val_loss:.2f}}.h5")

    # ModelCheckpoint callback to save the best model based on validation loss
    checkpoint = ModelCheckpoint(
        model_filepath, 
        monitor="val_loss", 
        save_best_only=True, 
        verbose=1
    )

    # TensorBoard callback for visualization
    tb_callback = TensorBoard(log_dir=tb_log_dir)

    return checkpoint, tb_callback


# In[76]:


def plot_history(history, model_name):    
    directory = 'literatures/figures/' + model_name 
    if not os.path.exists(directory):
        os.makedirs(directory)

    # --- Loss Plot ---
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], color='teal', label='training_loss')
    ax.plot(history.history['val_loss'], color='orange', label='val_loss')
    plt.ylim(0, 5)
    fig.suptitle('Loss', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{directory}/loss.jpg', dpi=500, bbox_inches='tight')

    # --- Accuracy Plot ---
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], color='teal', label='training_accuracy')
    ax.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    plt.ylim(0, 1)
    fig.suptitle('Accuracy', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{directory}/accuracy.jpg', dpi=500, bbox_inches='tight')


# In[77]:


def test_model(model, test_generator, model_name):
    y_true = test_generator.classes
    yhat = model.predict(test_generator)
    y_pred = [np.argmax(element) for element in yhat]
    cm = metrics.confusion_matrix(y_true,y_pred)
    # Print confusion matrix
    print("Confusion Matrix:")
    print(cm)
    accuracy = metrics.accuracy_score(y_true,y_pred)
    precision = metrics.precision_score(y_true,y_pred, average='macro')
    recall = metrics.recall_score(y_true,y_pred, average='macro')
    f1 = metrics.f1_score(y_true,y_pred, average='macro')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Generate classification report
    report = classification_report(y_true, y_pred)
    print("Classification report", report)
    
    # Display confusion matrix
    
    directory = 'literatures/figures/' + model_name 
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    class_labels = ['mild_damage', 'minimal_or_noDamage', 'severelyDamage_or_dead']
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    # Plot the confusion matrix
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 22})
    cm_display.plot(cmap='Blues', ax=ax)
    plt.xticks(rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    # Display the plot
    #plt.show()
    plt.savefig('literatures/figures/' + model_name + '/cm.jpg',dpi=500, bbox_inches='tight')
    
    # Measure average inference time over multiple runs
    '''num_runs = 500
    total_time = 0.0

    for _ in range(num_runs):
        start_time = time.time()
        yhat = model.predict(test_generator)
        y_pred = [np.argmax(element) for element in yhat]
        end_time = time.time()
        total_time += (end_time - start_time)

    average_inference_time = total_time / num_runs
    print(f"Average inference time over {num_runs} runs: {average_inference_time:.6f} seconds")'''


# In[ ]:




