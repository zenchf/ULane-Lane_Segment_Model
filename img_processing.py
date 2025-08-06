import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from carvana_dataset import CarvanaDataset

# Verisetini yükleyin
DATA_PATH = "/home/nvidia/Desktop/zece/GASimple"
dataset = CarvanaDataset(DATA_PATH)

# Rastgele bir örnek al
random_index = 42  # Burada rastgele bir index belirleyebilirsiniz
img, mask = dataset[random_index]

# Boyutları yazdır
print(f"Image size: {img.shape}")
print(f"Mask size: {mask.shape}")

# Resmi görselleştirme
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Orijinal Resim
ax[0].imshow(img.permute(1, 2, 0))  # Tensor'ü görselleştirmek için permute ile boyutları değiştiriyoruz
ax[0].set_title("Transformed Image")
ax[0].axis('off')

# Maske
ax[1].imshow(mask[0], cmap='gray')  # Maskeyi gösterirken tek kanal olduğu için cmap='gray' kullanıyoruz
ax[1].set_title("Mask")
ax[1].axis('off')

plt.show()

