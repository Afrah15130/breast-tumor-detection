# train.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from models import HybridCNNGNN
from utils import crop_two_breasts, extract_tumor_region, superpixel_segmentation, build_superpixel_graph, safe_save_image

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Data Augmentation ----------------
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------- Dataset ----------------
class BreastDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_two_breasts(img)
        img = extract_tumor_region(img)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

# ---------------- Load Dataset ----------------
healthy_dir = "dataset/healthy"
unhealthy_dir = "dataset/unhealthy"

healthy_paths = [os.path.join(healthy_dir,f) for f in os.listdir(healthy_dir)]
unhealthy_paths = [os.path.join(unhealthy_dir,f) for f in os.listdir(unhealthy_dir)]

image_paths = healthy_paths + unhealthy_paths
labels = [0]*len(healthy_paths) + [1]*len(unhealthy_paths)

X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

train_dataset = BreastDataset(X_train, y_train, transform=train_transforms)
test_dataset = BreastDataset(X_test, y_test, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ---------------- Model ----------------
model = HybridCNNGNN(use_gnn=False, trustnet=True).to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------- Training ----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# ---------------- Evaluation ----------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print("Final Accuracy:", correct/total)

torch.save(model.state_dict(), "cnn_tumor_contour_model.pth")
print("Model saved as 'cnn_tumor_contour_model.pth'")