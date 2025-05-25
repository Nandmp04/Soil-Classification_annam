import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Load the trained model ---
model = load_model('/content/soil_model.h5')
print("✅ Model loaded.")

# --- Constants ---
IMG_H, IMG_W = 128, 128
soil_types = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']
label_map = {i: s for i, s in enumerate(soil_types)}

# --- Test images folder ---
test_folder = '/content/drive/MyDrive/soil_classification/test'

# --- Prediction function ---
def predict_soil_type(image_path):
    img = load_img(image_path, target_size=(IMG_H, IMG_W))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = np.argmax(model.predict(arr, verbose=0), axis=1)[0]
    return label_map[pred]

# --- Predict and store results ---
results = []

for filename in sorted(os.listdir(test_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png','.gif','.webp')):
        path = os.path.join(test_folder, filename)
        pred = predict_soil_type(path)
        results.append({'image_id': filename, 'soil_type': pred})

# --- Save to CSV ---
output_df = pd.DataFrame(results)
csv_output_path = '/content/predictions.csv'
output_df.to_csv(csv_output_path, index=False)

print(f"✅ Predictions saved to {csv_output_path}")
