import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Define Autoencoder Model
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

# Load trained model
device = torch.device("cpu")
model = Autoencoder().to(device)


model_path = r"D:/spacedebris/autoencoder_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define Image Transformation
transform = transforms.Compose([transforms.ToTensor()])

def compute_metrics(original, reconstructed):
    """Compute MSE and SSIM."""
    mse = np.mean((original - reconstructed) ** 2)
    
    
    min_dim = min(original.shape[:2])
    win_size = 7 if min_dim >= 7 else min_dim 
    
    ssim_score = ssim(original, reconstructed, channel_axis=-1, data_range=255, win_size=win_size)
    
    return mse, ssim_score


image_dir = r"D:/DL/test_images/*.png"  
image_paths = glob.glob(image_dir)

if not image_paths:
    raise FileNotFoundError(f"No images found in {image_dir}")

mse_list, ssim_list = [], []

# Evaluate Autoencoder
for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"Warning: File {img_path} not found. Skipping...")
        continue

    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Unable to load {img_path}. Skipping...")
        continue

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed_tensor = model(img_tensor)

    reconstructed_img = (reconstructed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    mse, ssim_score = compute_metrics(img, reconstructed_img)
    mse_list.append(mse)
    ssim_list.append(ssim_score)

    print(f"{img_path}: MSE={mse:.4f}, SSIM={ssim_score:.4f}")

#to print vals
if mse_list and ssim_list:
    print(f"\nFinal Model Evaluation:")
    print(f"Average MSE: {np.mean(mse_list):.4f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
else:
    print("No valid images were processed.")
