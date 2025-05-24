1.Soil Type Classification using CNN
This project uses a Convolutional Neural Network (CNN) to classify soil images into four categories: Alluvial, Black, Clay, and Red soil. The model is built and trained using TensorFlow/Keras on labeled image data.

2.Dataset Desciption
soil_classification/
│
├── train_labels.csv             # CSV with image_id and soil_type
├── train/                       # Folder with training images
├── test/                        # Folder with test images (for inference)
├── soil_model.h5               # Saved trained model

3. Dataset Format
train_labels.csv: A CSV with two columns:

image_id – Filename of the image (e.g., img_123.jpg)

soil_type – One of the four classes: alluvial, black, clay, red

train/: A folder containing all training images referenced in the CSV.

4.Model Architecture
A simple sequential CNN:

Conv2D(32) → MaxPooling2D  
→ Conv2D(64) → MaxPooling2D  
→ Conv2D(128) → MaxPooling2D  
→ Flatten → Dense(128) → Dropout(0.5) → Dense(4, softmax)

