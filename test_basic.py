import yaml
import os
import glob

def test_yaml_paths():
    """Тест 1: Проверка, что пути в YAML существуют"""
    print("Запуск Теста 1 (Пути в YAML)...")
    with open('configs/ppe.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = config['path'].replace('../', '') 
    assert os.path.exists(os.path.join(base_path, config['train'])), "Train path not found"
    assert os.path.exists(os.path.join(base_path, config['val'])), "Val path not found"
    assert os.path.exists(os.path.join(base_path, config['test'])), "Test path not found"
    print("✅ Тест 1 пройден: Пути в YAML валидны\n")

def test_labels_range():
    """Тест 2: Проверка, что координаты в разметке лежат в [0, 1]"""
    print("Запуск Теста 2 (Диапазон координат)...")
    label_files = glob.glob('data_clean/test/labels/*.txt')[:10] 
    checked_boxes = 0
    for lf in label_files:
        with open(lf, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    coords = [float(x) for x in parts[1:]]
                    for c in coords:
                        assert 0.0 <= c <= 1.0, f"Координата {c} вне [0, 1] в файле {lf}"
                    checked_boxes += 1
    print(f"✅ Тест 2 пройден: Проверено {checked_boxes} рамок, все координаты в норме\n")

if __name__ == '__main__':
    test_yaml_paths()
    test_labels_range()
    print("--- Все минимальные тесты успешно пройдены! ---")