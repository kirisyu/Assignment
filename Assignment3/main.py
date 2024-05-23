import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

# 이미지 로드
image = Image.open('sample.png').convert('RGB')
print("Image 패치 형태:", image.size)

# ViT 전처리 함수
def preprocess_vit(image):
    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    vit_image = vit_transform(image)

    # 이미지 패치 분할
    patch_size = 16
    num_patches = (224 // patch_size) ** 2
    patches = vit_image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    patches = patches.reshape(num_patches, -1)

    return patches

# CNN 전처리 함수
def preprocess_cnn(image):
    cnn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    cnn_image = cnn_transform(image)

    return cnn_image

# ViT 전처리 적용
vit_patches = preprocess_vit(image)
print("ViT 패치 형태:", vit_patches.shape)

# CNN 전처리 적용
cnn_image = preprocess_cnn(image)
print("CNN 이미지 형태:", cnn_image.shape)
