import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

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

# Load Trained Model
device = torch.device("cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load("D:/spacedebris/autoencoder_model.pth", map_location=device))
model.eval()

# Define Image Transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

def compute_reconstruction_error(image_path):
    """ Computes reconstruction error for a given image. """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = model(img_tensor)
    
    error = torch.mean((img_tensor - reconstructed) ** 2).item()
    return error

# Load JWST PNG Images
image_paths = glob.glob("D:/DL/jwstimg/RGB_Converted/*.png")  # Update path for PNG images

# Compute Anomaly Scores
anomaly_scores = {}
for img_path in image_paths:
    error = compute_reconstruction_error(img_path)
    anomaly_scores[img_path] = error
    print(f"{img_path}: Reconstruction Error = {error:.6f}")

# Sort images by anomaly score
sorted_anomalies = sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)

# Plot Anomaly Scores
plt.figure(figsize=(10, 5))
plt.bar([x[0].split("/")[-1] for x in sorted_anomalies], [x[1] for x in sorted_anomalies], color='blue')
plt.xlabel("Image Name")
plt.ylabel("Reconstruction Error (Anomaly Score)")
plt.title("Anomaly Scores for JWST PNG Images")
plt.xticks(rotation=45, ha="right")
plt.show()
