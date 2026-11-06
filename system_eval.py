import os
import time
import glob
import psutil
import cv2
import numpy as np
from ultralytics import YOLO

def measure_system_metrics(model_path, exp_name):
    print(f"\n--- Системная оценка для: {exp_name} ---")
    process = psutil.Process(os.getpid())
    
    mem_before = process.memory_info().rss / (1024 * 1024) 
    model = YOLO(model_path)
    mem_after = process.memory_info().rss / (1024 * 1024)
    print(f"Память: До загрузки {mem_before:.1f} МБ, После загрузки {mem_after:.1f} МБ (Модель занимает: {mem_after - mem_before:.1f} МБ)")

    test_images = glob.glob('data_clean/test/images/*.jpg')[:10]
    if not test_images:
        print("Нет картинок для теста!")
        return

    # 1. Базовый прогон (Оригинал)
    start_time = time.time()
    orig_detections = 0
    orig_conf_sum = 0
    for img_path in test_images:
        res = model(img_path, verbose=False)[0]
        orig_detections += len(res.boxes)
        if len(res.boxes) > 0:
            orig_conf_sum += sum(res.boxes.conf).item()
            
    total_time = time.time() - start_time
    fps = len(test_images) / total_time
    avg_conf_orig = (orig_conf_sum / orig_detections) if orig_detections > 0 else 0
    print(f"Пропускная способность: {fps:.2f} FPS")
    print(f"Оригинал: Найдено объектов {orig_detections}, Средняя уверенность {avg_conf_orig:.2f}")

    # 2. Стресс-тест (Затемнение на 50%)
    dark_detections = 0
    dark_conf_sum = 0
    for img_path in test_images:
        img = cv2.imread(img_path)
        dark_img = cv2.convertScaleAbs(img, alpha=0.5, beta=0) 
        res = model(dark_img, verbose=False)[0]
        dark_detections += len(res.boxes)
        if len(res.boxes) > 0:
            dark_conf_sum += sum(res.boxes.conf).item()
            
    avg_conf_dark = (dark_conf_sum / dark_detections) if dark_detections > 0 else 0
    print(f"Стресс-тест (Тёмное): Найдено объектов {dark_detections}, Средняя уверенность {avg_conf_dark:.2f}")
    if orig_detections > 0:
        print(f"Падение количества детекций: {(1 - dark_detections/orig_detections)*100:.1f}%")

    # 3. Правило отказа (Fallback)
    print("\n[Проверка правила отказа]")
    critical_classes = [0, 2, 4, 7]
    rejected_count = 0
    for img_path in test_images:
        res = model(img_path, verbose=False)[0]
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in critical_classes and conf < 0.60:
                rejected_count += 1
                break 
                
    print(f"Кадров отправлено оператору (confidence < 0.6): {rejected_count} из {len(test_images)}")

if __name__ == '__main__':
    measure_system_metrics('runs/detect/runs/ppe/e1-opora-3/weights/best.pt', "E1 (imgsz=640)")