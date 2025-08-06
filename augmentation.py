import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import tqdm

# Paths to images and masks
image_dir = "/home/nvidia/Desktop/zece/GASimple/images"
mask_dir = "/home/nvidia/Desktop/zece/GASimple/masks"

# Output directories
aug_image_dir = "/home/nvidia/Desktop/zece/GASimple/images"
aug_mask_dir = "/home/nvidia/Desktop/zece/GASimple/masks"

# Create output directories if they don't exist
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# Define the data augmentation transformations
transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(10, 10)),
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.5, hue=0.10),
])

# Function to apply augmentation to both image and mask
def augment_image_and_mask(image_path, mask_path):
    # Read image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    # Apply the transformation
    augmented_image = transform(image)
    augmented_mask = transform(mask)
    
    return augmented_image, augmented_mask

# Loop through all images in the image directory
image_files = os.listdir(image_dir)
for img_name in tqdm.tqdm(image_files):
    # Get the corresponding mask file name
    mask_name = img_name
    image_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)
    
    # Check if the mask file exists
    if not os.path.exists(mask_path):
        print(f"Mask file for {img_name} not found. Skipping.")
        continue
    
    # Apply data augmentation
    augmented_image, augmented_mask = augment_image_and_mask(image_path, mask_path)
    
    # Generate new names for the augmented images and masks
    augmented_image_name = f"aug_{img_name}"
    augmented_mask_name = f"aug_{mask_name}"

    # Save augmented image and mask with new names
    augmented_image.save(os.path.join(aug_image_dir, augmented_image_name))
    augmented_mask.save(os.path.join(aug_mask_dir, augmented_mask_name))
    print("images saved {augmented_image_name}")

print("Data augmentation completed.")

