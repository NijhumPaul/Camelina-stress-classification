#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, Model

# ===== FIX ME if your model path moves =====
MODEL_PATH = r"custom/custom-20250809-123155-loss-0.11.h5"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 1

# MUST match the training class order (index 0,1,2,...)
IDX2LABEL = ["Mild", "Minimal/no damage", "severe/dead"]
IDX2SCORE = [1.5, 3.0, 0.0]   # optional scoring you use

def create_hybrid_model(input_shape=(224, 224, 3), num_classes=3):
    """Rebuilds your shallow DenseNet + tiny Transformer head (fallback)."""
    base = DenseNet121(include_top=False, input_shape=input_shape)
    # truncate at conv2_block6_concat as in your code
    base = Model(inputs=base.input, outputs=base.get_layer("conv2_block6_concat").output)

    inputs = layers.Input(shape=input_shape)
    x = base(inputs)                        # (H,W,C)
    x = layers.GlobalAveragePooling2D()(x)  # (C,)

    num_patches = 16
    c = x.shape[-1]
    if c % num_patches != 0:
        raise ValueError(f"Channels {c} not divisible by num_patches {num_patches}.")
    proj = c // num_patches
    x = layers.Reshape((num_patches, proj))(x)

    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=8, key_dim=proj)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)

def load_model_flex(path: str, input_shape=(224,224,3), num_classes=3) -> tf.keras.Model:
    """Try load full model, else rebuild and load weights."""
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"Loaded full model from: {path}")
        return m
    except Exception as e:
        print(f"load_model failed ({e}); rebuilding architecture and loading weights…")
        m = create_hybrid_model(input_shape=input_shape, num_classes=num_classes)
        m.load_weights(path)
        print(f"Loaded weights into rebuilt model from: {path}")
        return m

def main():
    ap = argparse.ArgumentParser(description="Predict CSV from an image folder.")
    ap.add_argument("image_folder", help="Folder of images (recurses into subfolders).")
    ap.add_argument("output_csv",   help="Output CSV path, e.g., AIscore.csv")
    args = ap.parse_args()

    # collect images recursively
    img_paths = []
    for r, _, files in os.walk(args.image_folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_paths.append(os.path.join(r, f))
    if not img_paths:
        print(f"No images found under: {args.image_folder}")
        sys.exit(1)

    df = pd.DataFrame({
        "filepath": img_paths,
        "filename": [os.path.splitext(os.path.basename(p))[0].lower() for p in img_paths],
    })

    # load model
    model = load_model_flex(MODEL_PATH, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                            num_classes=len(IDX2LABEL))
    # sanity: last units vs label count
    last = model.layers[-1]
    units = getattr(last, "units", None)
    if units and units != len(IDX2LABEL):
        print(f"WARNING: model outputs {units} classes but IDX2LABEL has {len(IDX2LABEL)}.")

    # preprocessing EXACTLY like your 94% eval
    pre_in_densenet = tf.keras.applications.densenet.preprocess_input
    test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=pre_in_densenet)

    gen = test_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filepath",
        y_col=None,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode=None,
        shuffle=False,
        batch_size=BATCH_SIZE,
    )

    preds = model.predict(gen, verbose=1)
    if preds.ndim == 1:
        preds = preds[:, None]
    y_idx = np.argmax(preds, axis=1)

    labels = [IDX2LABEL[i] if i < len(IDX2LABEL) else f"class_{i}" for i in y_idx]
    scores = [IDX2SCORE[i] if i < len(IDX2SCORE) else None for i in y_idx]

    out = pd.DataFrame({
        "filename": df["filename"],
        "class_label": labels,
        "class_score": scores,
    })

    # ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out.to_csv(args.output_csv, index=False)
    print(f"✅ Saved {len(out)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()
