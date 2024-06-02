import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# # Random Crop % Random 좌우반전
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
#     transforms.ConvertImageDtype(torch.float32)
# ])

# # Random Crop % Random 밝기 변화
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.ConvertImageDtype(torch.float32)
# ])

# Random Crop % Random 회전
transform = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])



data = ImageFolder(root="data/train")

output_dir = "rotated_data/train"
os.makedirs(output_dir, exist_ok=True)

for idx, (img, label) in enumerate(data):
    transformed_img = transform(img)

    pil_img = transforms.ToPILImage()(transformed_img)

    original_img_path = data.imgs[idx][0]
    relative_path = os.path.relpath(original_img_path, "data/train")

    new_img_path = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(new_img_path), exist_ok=True)

    pil_img.save(new_img_path)

print("Transformation and saving completed.")
