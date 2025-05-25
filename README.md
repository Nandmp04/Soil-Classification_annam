**Challenge-1**

Image Classification using CNN
This project demonstrates how to train and evaluate a Convolutional Neural Network (CNN) for image classification using TensorFlow/Keras. The model is designed to learn from labeled image data and predict the class of unseen images.

**Requirements**
Install the required packages using:

bash
Copy
Edit
pip install tensorflow pandas numpy scikit-learn
Optional (for model architecture visualization):

bash
Copy
Edit
pip install pydot
apt-get install -y graphviz
**Training the Model**
To train the model:

bash
Copy
Edit
python train_model.py
This script will:

Read image paths and labels from train_labels.csv

Preprocess images (resizing, normalization)

Train a CNN on the dataset

Save the trained model as model.h5

**Making Predictions**
To make predictions on new images:

bash
Copy
Edit
python predict.py
This script will:

**Load the trained model**

Process images from the test/ directory

Predict labels

Save the results to predictions.csv

**Challenge-2**

This project performs unsupervised image classification using feature extraction from a pre-trained deep learning model followed by clustering (e.g., KMeans).

**How to Run**

**Install dependencies**

bash
Copy
Edit
pip install -r requirements.txt
Prepare the image folder
Place all images to be processed in a directory, for example, test/.

**Run the script**

bash
Copy
Edit
python your_script_name.py
View the output
A CSV file will be generated with two columns:

image_id: name of the input image

label: cluster assignment (e.g., 0 or 1)
