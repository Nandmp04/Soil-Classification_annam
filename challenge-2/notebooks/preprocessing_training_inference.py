import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Paths ---
TEST_FOLDER = "/content/drive/MyDrive/soil-classification-part-2/soil_competition-2025/test"
CSV_OUTPUT_PATH = "/content/clustered_soil_predictions.csv"

# --- Image transform for pre-trained model ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load pretrained ResNet18 model as feature extractor ---
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()  # remove classification head
resnet.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

# --- Feature extraction ---
features = []
paths = []

print("📦 Extracting features from images...")
for fname in tqdm(sorted(os.listdir(TEST_FOLDER))):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
        image_path = os.path.join(TEST_FOLDER, fname)
        try:
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = resnet(img_tensor).cpu().numpy().flatten()
            features.append(feat)
            paths.append(fname)
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

features = np.array(features)

# --- Dimensionality Reduction (optional) ---
print("📉 Reducing dimensionality using PCA...")
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(features)

# --- KMeans Clustering (k=2) ---
print("🧠 Clustering images into 2 groups...")
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_features)

# --- Determine which cluster is soil (assume larger cluster is soil) ---
counts = np.bincount(cluster_labels)
soil_cluster = np.argmax(counts)
binary_labels = [1 if label == soil_cluster else 0 for label in cluster_labels]

# --- Save results to CSV ---
df = pd.DataFrame({
    "image_id": paths,
    "label": binary_labels
})
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"✅ Saved clustered results to: {CSV_OUTPUT_PATH}")
