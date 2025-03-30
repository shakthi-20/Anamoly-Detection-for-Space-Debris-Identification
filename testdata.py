import os
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import random_split


dataset_path = "D:/d_data_filtered"
test_save_path = "D:/DL/test_images/"


if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Check folder structure.")

# Define transformation
transform = transforms.Compose([transforms.ToTensor()])

# Load 
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# 80/20
train_ratio = 0.8
train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size
_, test_dataset = random_split(full_dataset, [train_size, test_size])  

# test folder
os.makedirs(test_save_path, exist_ok=True)

# Save test images
for idx, (image, _) in enumerate(test_dataset):
    save_image(image, os.path.join(test_save_path, f"test_image_{idx}.png"))

print(f"Saved {len(test_dataset)} test images to {test_save_path}")
