import cv2
import torch
from torchvision import transforms
from unet import UNet
from PIL import Image
import numpy as np

def process_video(video_path, model_pth, output_path, device):
    # Modeli yükle
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()

    # Video okuyucu ve yazıcı ayarları
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # Görüntü dönüşümleri
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Orijinal frame'i işleme
        img = transform(frame).float().to(device)
        img = img.unsqueeze(0)  # (B, C, H, W)

        # Model tahmini
        with torch.no_grad():
            pred_mask = model(img)

        # Tahmin maskesini işle
        pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
        pred_mask = np.squeeze(pred_mask)  # (H, W)
        pred_mask = (pred_mask > 0).astype(np.uint8) * 255  # İkili maske (0 veya 255)

        # Maskeyi orijinal kare ile birleştir
        pred_mask = cv2.resize(pred_mask, (frame_width, frame_height))
        color_mask = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
        combined_frame = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)

        # Sonuç videoya yaz
        out.write(combined_frame)

        # Görüntülemek için (isteğe bağlı)
        cv2.imshow('Segmented Video', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO_PATH = "./data/videos/yarisma.mkv"  # Giriş videosu yolu
    MODEL_PATH = "./backup.pth_epoch10.pth"  # Eğitilmiş modelin yolu
    OUTPUT_PATH = "./data/output-videos/yarisma_yetis.avi"  # Çıkış videosu yolu

    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_video(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH, device)

