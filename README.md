**Challenge-2**

This project performs unsupervised image classification using feature extraction from a pre-trained deep learning model followed by clustering (e.g., KMeans).

**How to Run**

Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Prepare the image folder
Place all images to be processed in a directory, for example, test/.

Run the script

bash
Copy
Edit
python your_script_name.py
View the output
A CSV file will be generated with two columns:

image_id: name of the input image

label: cluster assignment (e.g., 0 or 1)
