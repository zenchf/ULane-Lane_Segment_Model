import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from carvana_dataset import CarvanaDataset

def dice_score(y_pred, y_true, threshold=0.5):
    """Calculate Dice Score."""
    y_pred = torch.sigmoid(y_pred)  # Apply sigmoid to get probabilities
    y_pred = (y_pred > threshold).float()  # Binarize predictions
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2. * intersection) / (union + 1e-7)
    return dice.item()

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16
    EPOCHS = 40
    DATA_PATH = "/home/nvidia/Desktop/zece/GASimple"
    MODEL_SAVE_PATH = "/home/nvidia/Desktop/zece/backup.pth"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    train_dataset = CarvanaDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        val_dice_score = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()

                # Dice Score Calculation
                val_dice_score += dice_score(y_pred, mask)

            val_loss = val_running_loss / (idx + 1)
            avg_dice_score = val_dice_score / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print(f"Dice Score EPOCH {epoch+1}: {avg_dice_score:.4f}")
        print("-"*30)
        
        save_path = f"{MODEL_SAVE_PATH}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    	#torch.save(model.state_dict(), MODEL_SAVE_PATH)
        torch.cuda.empty_cache()  # Her epoch sonunda bellek temizlemek için
    	

