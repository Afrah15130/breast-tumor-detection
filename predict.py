# predict.py
import os
import cv2
import torch
from torchvision import transforms
from utils import crop_two_breasts, extract_tumor_region, safe_save_image
from models import HybridCNNGNN
from matplotlib import pyplot as plt

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_FOLDER = "dataset/unhealthy"

# ---------------- LOAD MODEL ----------------
model = HybridCNNGNN(use_gnn=False, trustnet=True).to(DEVICE)
model.load_state_dict(torch.load("cnn_tumor_contour_model.pth", map_location=DEVICE))
model.eval()

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------- CHOOSE IMAGE ----------------
all_files = [f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith((".bmp",".jpg",".png"))]
print("Select image to predict:")
for idx,f in enumerate(all_files):
    print(f"{idx}: {f}")
choice = int(input("Enter index: "))
img_path = os.path.join(DATASET_FOLDER, all_files[choice])

# ---------------- PROCESS IMAGE ----------------
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_crop = crop_two_breasts(img)
tumor_img = extract_tumor_region(img_crop)

# ---------------- PREDICT ----------------
img_tensor = test_transforms(tumor_img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    output = model(img_tensor)
    pred = torch.sigmoid(output) > 0.5

# ---------------- SAVE & SHOW ----------------
safe_save_image(tumor_img, "predicted_tumor.jpg")
print(f"Prediction: {'Unhealthy' if pred.item() else 'Healthy'}")
print("Tumor image saved as 'predicted_tumor.jpg'")

# Show inline
plt.imshow(tumor_img)
plt.title(f"Prediction: {'Unhealthy' if pred.item() else 'Healthy'}")
plt.axis('off')
plt.show()