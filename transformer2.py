#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install --upgrade keras
#!pip install ops
#!pip install -U imbalanced-learn


# In[1]:


import os
import tensorflow as tf
os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]
#
import keras
from keras import layers
from keras import ops

import numpy as np
import matplotlib.pyplot as plt
import utils
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import ADASYN

# In[2]:


#!pip freeze


# In[3]:


learning_rate = 0.001
weight_decay = 0.0001
batch_size = 16
num_epochs = 700  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size
input_shape = (32, 32, 3)
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 1
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier
num_classes=3


# In[4]:


# Path to the root directory containing class folders
data_dir = "split_2_5_25"
models_name="ViT"

# Parameters
#img_size = (224, 224)      # Target image size (width, height)
classes = os.listdir(data_dir)  # Automatically detect class names
class_indices = {cls: idx for idx, cls in enumerate(classes)}  # Map class names to indices

# Initialize data and labels lists
x = []
y = []

# Load images and labels
for cls in classes:
    cls_dir = os.path.join(data_dir, cls)
    for img_name in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, img_name)
        img = load_img(img_path, target_size=input_shape)  # Load image and resize
        img_array = img_to_array(img)                  # Convert to numpy array
        x.append(img_array)                            # Add to data list
        y.append(class_indices[cls])                  # Add class label

# Convert to numpy arrays
x = np.array(x, dtype="float32")
y = np.array(y, dtype="int")

# Normalize image data to [0, 1]
#x /= 255.0

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# Print shapes
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


# In[6]:

unique, counts = np.unique(y_train, return_counts=True)
print("class distribution:")
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples")



# Flatten images for ADASYN
#x_train_flat = x_train.reshape(x_train.shape[0], -1)

# Apply ADASYN
#adasyn = ADASYN(sampling_strategy='minority', random_state=42)
#x_train_adasyn, y_train_adasyn = adasyn.fit_resample(x_train_flat, y_train)

# Reshape back to original image dimensions
#x_train_adasyn = x_train_adasyn.reshape(-1, *x_train.shape[1:])
#print("New dataset shape:", x_train_adasyn.shape, y_train_adasyn.shape)


# In[8]:


#unique, counts = np.unique(y_train_adasyn, return_counts=True)
#print("Balanced class distribution:")
#for cls, count in zip(unique, counts):
#    print(f"Class {cls}: {count} samples")


# In[9]:


data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        layers.RandomShear(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)


# In[10]:


#Implement multilayer perceptron (MLP)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# In[11]:


#Implement patch creation as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


# In[12]:


plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")


# In[13]:


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


# In[14]:


def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    model.summary()
    return model


# In[ ]:


def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "checkpoint/vit/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,   # Stop if no improvement after 50 epochs
    restore_best_weights=True
    )


    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.15,
        callbacks=[checkpoint_callback,early_stopping],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    print(model.summary)

    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)


def plot_history(history, model_name):
    directory = 'literatures/figures/' + models_name 
    # Create the directory if it does not exist
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

utils.plot_history(history , models_name)
import time

# Select a random image from x_test
sample_input = np.expand_dims(x_test[0], axis=0)  # Single image with batch dimension

# Run inference 500 times and take the average
num_runs = 500
total_time = 0.0

for _ in range(num_runs):
    start_time = time.time()
    _ = vit_classifier.predict(sample_input, verbose=0)  # Run inference
    end_time = time.time()
    total_time += (end_time - start_time)

# Compute average inference time per image
average_inference_time = (total_time / num_runs) * 1000  # Convert to milliseconds
print(f"Average inference time for a single image (over {num_runs} runs): {average_inference_time:.4f} ms")

# Measure batch inference time (for the whole test set)
start_time = time.time()
_ = vit_classifier.predict(x_test, verbose=0)  # Run inference on the whole test set
end_time = time.time()

# Compute batch inference time
batch_inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
average_time_per_image = batch_inference_time / len(x_test)

print(f"Total inference time for {len(x_test)} images: {batch_inference_time:.4f} ms")
print(f"Average inference time per image (batch mode): {average_time_per_image:.4f} ms")

#plot_history("loss")
#plot_history("top-5-accuracy")

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Get model predictions on test set
y_pred_probs = vit_classifier.predict(x_test)  # Predict probabilities
y_pred_labels = np.argmax(y_pred_probs, axis=1)  # Convert to class labels
# Compute Accuracy, Precision, Recall, and F1-score
accuracy = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels, average='weighted')  # Use 'weighted' for multi-class
recall = recall_score(y_test, y_pred_labels, average='weighted')
f1 = f1_score(y_test, y_pred_labels, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Print detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))

# Display confusion matrix
    
directory = 'literatures/figures/' + models_name 
    # Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

cm = metrics.confusion_matrix(y_test, y_pred_labels)        
class_labels = ['mild_damage', 'minimal_or_noDamage', 'severelyDamage_or_dead']
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
# Plot the confusion matrix
fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 22})
cm_display.plot(cmap='Blues', ax=ax)
plt.xticks(rotation=90)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)
# Display the plot
#plt.show()
plt.savefig('literatures/figures/' + models_name + '/cm.jpg',dpi=500, bbox_inches='tight')

import time

num_runs = 5
total_time = 0.0

print(f"Running inference {num_runs} times to average inference time...")

for _ in range(num_runs):
    start_time = time.time()
    _ = vit_classifier.predict(x_test, batch_size=1, verbose=0)
    end_time = time.time()
    total_time += (end_time - start_time)

average_inference_time = total_time / num_runs
print(f"Average inference time for all test samples over {num_runs} runs: {average_inference_time:.4f} seconds")
print(f"Average inference time per sample: {average_inference_time / len(x_test):.6f} seconds")

#utils.test_model(vit_classifier, test_data_gen, models_name)




