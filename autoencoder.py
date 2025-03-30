import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Autoencoder Model
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

#  after img process data
dataset_path = "D:/d_data_filtered"

if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Check your folder structure")

transform = transforms.Compose([
    transforms.ToTensor()
])

full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Train-Test Split (80% train, 20% test)
train_ratio = 0.8
train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

#  DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train Autoencoder
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10  
loss_history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "autoencoder_model.pth")
print("Training completed & model saved!")

#  Evaluate Test Data
model.eval()
test_loss = 0

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Reconstruction Loss Over Epochs")
plt.grid()
plt.show()

# Reconstructed Imgs
def imshow(img):
    img = img.cpu().numpy().transpose((1, 2, 0))  # Convert tensor to NumPy
    img = np.clip(img, 0, 1)  # Clip pixel values
    plt.imshow(img)
    plt.axis('off')

data_iter = iter(test_loader)  # to check on test
images, _ = next(data_iter)
images = images.to(device)

# Get reconstructions
model.eval()
with torch.no_grad():
    reconstructed = model(images)

# original and reconstructed img
fig, axes = plt.subplots(2, 6, figsize=(10, 4))
for i in range(6):
    axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))  # Original
    axes[0, i].axis("off")
    axes[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0))  # Reconstructed
    axes[1, i].axis("off")

axes[0, 0].set_title("Original Images")
axes[1, 0].set_title("Reconstructed Images")
plt.show()
