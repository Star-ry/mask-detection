import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics

# transforms for image augmentation
transform = transforms.Compose([
    transforms.Resize(size=(112, 112)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

test_path = "DATA/data/test"

# Load test data
test_data = torchvision.datasets.ImageFolder(
    root=test_path, 
    transform=transform
)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2)

# Load the saved model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('weights/best_model.pth')
model = model.to(device)

# Evaluation
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        labels = labels.unsqueeze(1)

        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate metrics
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

accuracy = metrics.accuracy_score(all_labels, all_preds)
precision = metrics.precision_score(all_labels, all_preds)
recall = metrics.recall_score(all_labels, all_preds)
f1_score = metrics.f1_score(all_labels, all_preds)
roc_auc = metrics.roc_auc_score(all_labels, all_preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

# Plot ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_preds)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


import cv2
import os
from PIL import Image, ImageDraw

# Define directories
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the function to add border to an image
def add_border(image, color, border_width=10):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for i in range(border_width):
        draw.rectangle([i, i, width-i-1, height-i-1], outline=color)
    return image

# Load and process images
all_labels = []
all_preds = []

model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        labels = labels.unsqueeze(1)

        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        # Process and save images with borders
        for j in range(inputs.size(0)):
            img_tensor = inputs[j].cpu()
            img_pil = transforms.ToPILImage()(img_tensor)
            pred = preds[j].item()
            true_label = labels[j].item()

            if pred == true_label:
                color = 'green'
            else:
                color = 'red'

            img_with_border = add_border(img_pil, color)
            img_with_border.save(f"{output_dir}/image_{i*16 + j}.png")

# Create video from images
image_files = sorted([os.path.join(output_dir, img) for img in os.listdir(output_dir) if img.endswith('.png')])
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, (width, height))

for image_file in image_files:
    video.write(cv2.imread(image_file))

video.release()
