import os
import random
import glob
from ultralytics import YOLO
import cv2

# Загружаем предобученную модель (она лежит в папке models)
model = YOLO('models/yolov8n.pt')

# Берем 10 случайных картинок из нашего нового тестового набора
test_images = glob.glob('data_clean/test/images/*.jpg')
random.seed(42)
sample_images = random.sample(test_images, 10)

# Создаем папку для сохранения результатов E0
os.makedirs('results/e0_predictions', exist_ok=True)

print("--- РЕЗУЛЬТАТЫ E0 (Предобученная YOLOv8n на COCO) ---")
for img_path in sample_images:
    # Делаем предсказание
    results = model(img_path)
    
    # Получаем имена классов, которые нашла модель
    names = results[0].names
    boxes = results[0].boxes
    detected_classes = set()
    
    if boxes is not None:
        for cls_id in boxes.cls:
            detected_classes.add(names[int(cls_id)])
    
    print(f"Файл: {os.path.basename(img_path)}")
    if detected_classes:
        print(f"  Найдено (COCO-классы): {list(detected_classes)}")
    else:
        print("  Ничего не найдено")
        
    # Сохраняем картинку с нарисованными рамками
    save_path = os.path.join('results/e0_predictions', os.path.basename(img_path))
    results[0].save(filename=save_path)

print("\nСохраненные картинки с рамками лежат в: results/e0_predictions/")