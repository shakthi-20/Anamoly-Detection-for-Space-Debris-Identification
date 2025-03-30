import os
import cv2
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# (Bilateral + CLAHE + Compression)
input_folder = "D:/dldata"  
output_folder = "D:/d_data_filtered"  # Processed dataset folder

os.makedirs(output_folder, exist_ok=True)

def apply_filters(img):
    """ Apply Bilateral Filtering and CLAHE """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    return img

for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    output_category_path = os.path.join(output_folder, category)

    if os.path.isdir(category_path):
        os.makedirs(output_category_path, exist_ok=True)
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Cannot read {img_path}")
                    continue
                
                img = apply_filters(img)
                img = cv2.resize(img, (224, 224))  # Resize 

                output_path = os.path.join(output_category_path, img_name)
                success = cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 50])  # Compress

                if not success:
                    print(f"Failed to save: {output_path}")

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

print("Dataset filtering and compression completed")

# Load 
dataset_path = output_folder  # preprocessed used here

# Checking
if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Check your folder structure")

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize([0.5], [0.5])  
])

#  dataset and DataLoader
train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Dataset Properties
print("Classes:", train_dataset.classes)
print(f"Total Images: {len(train_dataset)}")

for images, labels in train_loader:
    print(f"Image batch shape: {images.shape}")  
    print(f"Label batch shape: {labels.shape}")  
    break  
