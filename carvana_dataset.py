import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CarvanaDataset(Dataset):
    def __init__(self, root_path, test=True):
        self.root_path = root_path
        if test:
            self.images = sorted([root_path+"/images/"+i for i in os.listdir(root_path+"/images/")])
            self.masks = sorted([root_path+"/masks/"+i for i in os.listdir(root_path+"/masks/")])
        else:
            self.images = sorted([root_path+"/images/"+i for i in os.listdir(root_path+"/images/")])
            self.masks = sorted([root_path+"/masks/"+i for i in os.listdir(root_path+"/masks/")])
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)
