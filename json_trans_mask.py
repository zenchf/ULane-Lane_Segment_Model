import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

# COCO JSON dosyasını oku
with open('coco_annotations.json', 'r') as f:
    coco_data = json.load(f)

# Resim boyutları (bu boyutlar, JSON'daki resimlerin boyutlarına uygun olmalı)
width = 640  # Gerekirse COCO JSON'dan alın
height = 480  # Gerekirse COCO JSON'dan alın

# Boş bir maske oluştur
mask = np.zeros((height, width), dtype=np.uint8)

# COCO'daki tüm segmentasyonları işlemeye başla
for annotation in coco_data['annotations']:
    segmentation = annotation['segmentation']  # Poligon segmentasyonu

    if isinstance(segmentation, list):  # Poligon segmentasyonu
        for poly in segmentation:
            poly = np.array(poly).reshape((len(poly) // 2, 2))  # Poligonu uygun şekle getirme
            poly = poly.astype(np.int32)
            
            # Poligonu çizme
            pil_image = Image.fromarray(mask)
            draw = ImageDraw.Draw(pil_image)
            draw.polygon([tuple(pt) for pt in poly], outline=1, fill=1)  # Beyaz renkle doldur
            mask = np.array(pil_image)
    
    elif isinstance(segmentation, dict):  # RLE formatında segmentasyon
        rle = segmentation
        mask_rle = coco_mask.decode(rle)  # RLE'yi çözme
        mask = np.maximum(mask, mask_rle)  # RLE maskesini ekleme

# Maskeyi kaydet
mask_image = Image.fromarray(mask * 255)  # 0-255 aralığına dönüştür
mask_image.save('output_mask.png')

