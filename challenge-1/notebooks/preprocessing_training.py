import pandas as pd
import os
import numpy as np

from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# 1. Paths
csv_path     = '/content/drive/MyDrive/soil_classification/train_labels.csv'
image_folder = '/content/drive/MyDrive/soil_classification/train'

# 2. Load labels CSV
df = pd.read_csv(csv_path)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(image_folder, x))

# 3. Encode soil_type → integer labels
soil_types = sorted(df['soil_type'].unique())   # ['alluvial','black','clay','red']
label_map  = {soil:i for i,soil in enumerate(soil_types)}
df['label'] = df['soil_type'].map(label_map)

# 4. Image preprocessing
IMG_H, IMG_W = 224, 224

def preprocess_image(path):
    img = load_img(path, target_size=(IMG_H, IMG_W))
    arr = img_to_array(img) / 255.0
    return arr

def build_dataset(df):
    X = np.stack([preprocess_image(p) for p in df['image_path']])
    y = df['label'].values
    return X, y

X, y = build_dataset(df)
y_cat = to_categorical(y, num_classes=len(soil_types))

# 5. Build model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_H, IMG_W, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(soil_types), activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train
model.fit(X, y_cat,
          epochs=35,
          batch_size=32,
          validation_split=0.1,
          verbose=2)

# 7. Save model
model.save('/content/soil_model.h5')
print("✅ Model saved at /content/soil_model.h5")
