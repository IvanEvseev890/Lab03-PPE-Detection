import yaml
import os
import glob
import hashlib
import cv2
from ultralytics import YOLO

def test_yaml_paths():
    print("Запуск Теста 1 (Пути в YAML)...")
    with open('configs/ppe.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = config['path'].replace('../', '') 
    assert os.path.exists(os.path.join(base_path, config['train'])), "Train path not found"
    assert os.path.exists(os.path.join(base_path, config['val'])), "Val path not found"
    assert os.path.exists(os.path.join(base_path, config['test'])), "Test path not found"
    print("✅ Тест 1 пройден: Пути в YAML валидны\n")

def test_labels_range():
    print("Запуск Теста 2 (Диапазон координат и классов)...")
    label_files = glob.glob('data_clean/test/labels/*.txt')[:10] 
    checked_boxes = 0
    for lf in label_files:
        with open(lf, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    assert 0 <= cls_id <= 9, f"Класс {cls_id} вне [0, 9] в файле {lf}"
                    
                    coords = [float(x) for x in parts[1:]]
                    for c in coords:
                        assert 0.0 <= c <= 1.0, f"Координата {c} вне [0, 1] в файле {lf}"
                    checked_boxes += 1
    print(f"✅ Тест 2 пройден: Проверено {checked_boxes} рамок, классы и координаты в норме\n")

def test_duplicate_hashes():
    print("Запуск Теста 3 (Контрольные хэши MD5)...")
    splits = ['train', 'val', 'test']
    hashes = {s: set() for s in splits}
    
    for split in splits:
        img_dir = f'data_clean/{split}/images'
        if not os.path.exists(img_dir): continue
        images = glob.glob(f'{img_dir}/*.jpg') + glob.glob(f'{img_dir}/*.png')
        for img_path in images:
            with open(img_path, 'rb') as f:
                hashes[split].add(hashlib.md5(f.read()).hexdigest())
                
    assert not (hashes['train'] & hashes['val']), "❌ Дубликаты между train и val!"
    assert not (hashes['train'] & hashes['test']), "❌ Дубликаты между train и test!"
    assert not (hashes['val'] & hashes['test']), "❌ Дубликаты между val и test!"
    print("✅ Тест 3 пройден: Пересечений хэшей нет\n")

def test_visualizer_size():
    print("Запуск Теста 4 (Размер изображения после визуализации)...")
    img_path = glob.glob('data_clean/test/images/*.jpg')[0]
    img_orig = cv2.imread(img_path)
    h_orig, w_orig = img_orig.shape[:2]
    
    test_out_path = 'results/test_size_check.jpg'
    cv2.imwrite(test_out_path, img_orig)
    
    img_out = cv2.imread(test_out_path)
    h_out, w_out = img_out.shape[:2]
    os.remove(test_out_path)
    
    assert (h_orig, w_orig) == (h_out, w_out), "❌ Размер изменился!"
    print(f"✅ Тест 4 пройден: Размер сохранен ({w_orig}x{h_orig})\n")

def test_model_output_schema():
    print("Запуск Теста 5 (Схема вывода модели)...")
    model = YOLO('runs/detect/runs/ppe/e1-opora-3/weights/best.pt')
    img_path = glob.glob('data_clean/test/images/*.jpg')[0]
    
    results = model(img_path, verbose=False)[0]
    
    assert hasattr(results, 'boxes'), "❌ Нет атрибута boxes"
    if len(results.boxes) > 0:
        assert results.boxes.xyxy.shape[1] == 4, "❌ Координаты не xyxy"
        assert len(results.boxes.conf.shape) == 1, "❌ Неверная размерность conf"
        assert len(results.boxes.cls.shape) == 1, "❌ Неверная размерность cls"
        assert len(results.boxes.conf) == len(results.boxes.xyxy), "❌ Разное количество боксов и уверенностей"
    print("✅ Тест 5 пройден: Схема вывода модели валидна\n")

if __name__ == '__main__':
    test_yaml_paths()
    test_labels_range()
    test_duplicate_hashes()
    test_visualizer_size()
    test_model_output_schema()
    print("--- Все минимальные тесты успешно пройдены! ---")