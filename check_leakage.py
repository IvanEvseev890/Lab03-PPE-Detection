import os
import re
from collections import defaultdict

# Пути к папкам (проверь, чтобы они совпадали с твоей структурой)
paths = {
    'train': 'data/train/images',
    'val': 'data/valid/images',
    'test': 'data/test/images'
}

def get_group_prefix(filename):
    # Пытаемся вытащить префикс из имени файла.
    name = os.path.splitext(filename)[0]
    
    # Ищем паттерны типа "xxx_frame" или "xxx_mp4"
    match = re.match(r'([a-zA-Z0-9]+)_(?:frame|mp4|jpg)', name)
    if match:
        return match.group(1)
    
    # Если паттерн не найден, берем первые 8 символов как псевдо-группу
    return name[:8]

groups = defaultdict(dict)

for split, path in paths.items():
    if not os.path.exists(path):
        print(f"⚠️ Путь не найден: {path}")
        continue
        
    files = os.listdir(path)
    for f in files:
        if f.endswith(('.jpg', '.jpeg', '.png')):
            prefix = get_group_prefix(f)
            groups[prefix][f] = split

# Проверяем пересечения
leaks = []
for prefix, files_dict in groups.items():
    splits_involved = set(files_dict.values())
    if len(splits_involved) > 1:
        leaks.append((prefix, files_dict))

print(f"\n--- АНАЛИЗ УТЕЧКИ ---")
print(f"Всего найдено уникальных групп (префиксов): {len(groups)}")

if leaks:
    print(f"❌ ОБНАРУЖЕНА УТЕЧКА! Найдено {len(leaks)} групп, которые есть в разных выборках.")
    print("Примеры утечек (первые 5):")
    for leak in leaks[:5]:
        print(f"  Группа '{leak[0]}' встречается в: {set(leak[1].values())}")
else:
    print("✅ Утечки не обнаружено! Префиксы файлов не пересекаются между train/val/test.")