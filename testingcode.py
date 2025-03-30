import os
import glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms

# Define Autoencoder Class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

# Load trained weights
model_path = "autoencoder_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully!")

# Define image transformation
transform = transforms.Compose([transforms.ToTensor()])

def compute_anomaly_score(image_path):
    """ Computes reconstruction error for a given image. """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    img_tensor = transform(img).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

    with torch.no_grad():
        reconstructed = model(img_tensor)

    error = torch.mean((img_tensor - reconstructed) ** 2).item()
    return error, img, reconstructed.cpu().squeeze().permute(1, 2, 0).numpy()

# Load test images
test_image_dir = r"D:/DL/test_images"  
anomaly_folder = r"D:/spacedebris/anomalies/"
os.makedirs(anomaly_folder, exist_ok=True)

test_images = glob.glob(os.path.join(test_image_dir, "*.png")) + glob.glob(os.path.join(test_image_dir, "*.webp"))

if not test_images:
    print("No test images found! Check the directory path.")
    exit()

# anomaly scores
anomaly_scores = {}
image_data = {}

for img_path in test_images:
    try:
        error, original, reconstructed = compute_anomaly_score(img_path)
        anomaly_scores[img_path] = error
        image_data[img_path] = (original, reconstructed)
        print(f"Image: {os.path.basename(img_path)}, Anomaly Score: {error:.6f}")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# dynamic anomaly threshold (Mean + 2*Std)
mean_score = np.mean(list(anomaly_scores.values()))
std_score = np.std(list(anomaly_scores.values()))
threshold = mean_score + 2 * std_score

# anomaly finding
anomalies = sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)  # Sort by highest error
top_5_anomalies = anomalies[:5]  #  top 5 anomalies

print(f"Anomaly threshold: {threshold:.6f}")
print(f"Detected {len([x for x in anomalies if x[1] > threshold])} anomalies.")

# sep folder
for img_path, score in anomalies:
    if score > threshold:
        os.rename(img_path, os.path.join(anomaly_folder, os.path.basename(img_path)))

print(f"Moved anomalies to: {anomaly_folder}")

# og vs reconstructed for top5
for i, (img_path, score) in enumerate(top_5_anomalies):
    original, reconstructed = image_data[img_path]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title(f"Original ({os.path.basename(img_path)})")
    axes[0].axis("off")

    axes[1].imshow(np.clip(reconstructed, 0, 1))
    axes[1].set_title(f"Reconstructed (Score: {score:.6f})")
    axes[1].axis("off")

    plt.show()
