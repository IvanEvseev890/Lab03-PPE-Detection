import time
import glob
import statistics
import numpy as np
from ultralytics import YOLO

def measure_latency(model_path, exp_name, num_images=50):
    print(f"\n--- Замер задержки для: {exp_name} ---")
    model = YOLO(model_path)
    
    # Берем 50 картинок из теста (достаточно для статистики)
    test_images = glob.glob('data_clean/test/images/*.jpg')[:num_images]
    
    if not test_images:
        print("Ошибка: нет тестовых изображений.")
        return

    # Прогрев (первый вызов всегда медленный)
    model(test_images[0], verbose=False)

    full_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    for img_path in test_images:
        # Замер полного вызова end-to-end
        start = time.perf_counter()
        results = model(img_path, verbose=False)
        end = time.perf_counter()
        
        full_times.append((end - start) * 1000) # переводим в мс
        
        # Вытаскиваем внутренние метрики Ultralytics (они уже в мс)
        speed = results[0].speed
        preprocess_times.append(speed['preprocess'])
        inference_times.append(speed['inference'])
        postprocess_times.append(speed['postprocess'])

    # Функция для расчета статистики
    def calc_stats(data, name):
        median = statistics.median(data)
        p95 = np.percentile(data, 95)
        print(f"  {name}: Медиана = {median:.2f} мс, p95 = {p95:.2f} мс")

    print(f"  Всего обработано изображений: {len(test_images)}")
    calc_stats(full_times, "Полный вызов (end-to-end)")
    print("  --- Внутренние этапы (по данным Ultralytics) ---")
    calc_stats(preprocess_times, "Препроцессинг (resize и т.д.)")
    calc_stats(inference_times, "Инференс (модель)")
    calc_stats(postprocess_times, "Постпроцессинг (NMS)")

if __name__ == '__main__':
    # Пути к лучшим весам E1 и E2
    e1_weights = 'runs/detect/runs/ppe/e1-opora-3/weights/best.pt'
    e2_weights = 'runs/detect/runs/ppe/e2-imgsz800/weights/best.pt'
    
    # Проверяем E1
    measure_latency(e1_weights, "E1 (imgsz=640)")
    
    # Проверяем E2
    measure_latency(e2_weights, "E2 (imgsz=800)")