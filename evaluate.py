from ultralytics import YOLO
import time
import glob
import sys

def evaluate_model(model_path, exp_name):
    print(f"\n--- Оценка модели: {exp_name} ---")
    model = YOLO(model_path)
    
    # 1. Оценка метрик на тестовой выборке (mAP, Precision, Recall)
    metrics = model.val(data='configs/ppe.yaml', split='test')
    
    print(f"\n[{exp_name}] mAP@0.5: {metrics.box.map50:.4f}")
    print(f"[{exp_name}] mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    # 2. Замер задержки (Latency)
    test_images = glob.glob('data_clean/test/images/*.jpg')[:20]
    if not test_images:
        print("Нет тестовых изображений для замера времени")
        return

    # Прогрев (первый вызов всегда медленный)
    model(test_images[0])
    
    start_time = time.time()
    for img in test_images:
        model(img)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / len(test_images)
    print(f"[{exp_name}] Средняя задержка инференса: {avg_time * 1000:.2f} мс\n")

if __name__ == '__main__':
    if len(sys.argv) > 2:
        evaluate_model(sys.argv[1], sys.argv[2])
    else:
        print("Использование: python evaluate.py <путь_к_весам> <название_эксперимента>")