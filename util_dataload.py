import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
# Define the transformation for the images
transform = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Create the dataset
full_dataset = datasets.ImageFolder(
   root="/home/sanjotst/img_cls/datasets/paddy-doctor-diseases-small-augmented-65k-split/train",
   transform=transform
)
# Calculate split sizes
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
# Generate random indices for the splits
dataset_indices = list(range(len(full_dataset)))
np.random.shuffle(dataset_indices)  # Shuffle the indices
train_indices, val_indices = dataset_indices[:train_size], dataset_indices[train_size:]
# Create Subsets using the indices
train_dataset = Subset(full_dataset, train_indices)
validation_dataset = Subset(full_dataset, val_indices)
# Create DataLoaders for the subsets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)
# Verify the data loading
# for images, labels in train_loader:
#    print(f"Batch of images shape: {images.shape}")
#    print(f"Batch of labels shape: {labels.shape}")
# print("this should not be printed")
# exit()  