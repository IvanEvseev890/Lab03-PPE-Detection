import os
import random
import shutil
from collections import defaultdict

# Пути к исходным данным (как сейчас в проекте)
input_dirs = {
    'train': 'data/train',
    'val': 'data/valid',
    'test': 'data/test'
}

# Выходная папка с чистым датасетом
output_dir = 'data_clean'

# Соотношение разбиения
split_ratio = {'train': 0.7, 'val': 0.15, 'test': 0.15}

# Сид для воспроизводимости
random.seed(42)

# Собираем все файлы по группам
groups = defaultdict(list)

for split, base_dir in input_dirs.items():
    img_dir = os.path.join(base_dir, 'images')
    lbl_dir = os.path.join(base_dir, 'labels')
    
    if not os.path.exists(img_dir):
        print(f"⚠️ Папка не найдена: {img_dir}")
        continue
        
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            # Группируем по первым 8 символам (как в проверке утечки)
            group_prefix = img_name[:8]
            
            # Проверяем наличие файла разметки
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            if os.path.exists(lbl_path):
                groups[group_prefix].append({
                    'img_src': os.path.join(img_dir, img_name),
                    'lbl_src': lbl_path,
                    'img_name': img_name,
                    'lbl_name': lbl_name
                })
            else:
                print(f"⚠️ Нет разметки для: {img_name}")

# Перемешиваем группы
group_keys = list(groups.keys())
random.shuffle(group_keys)

# Распределяем группы по выборкам
splits = {'train': [], 'val': [], 'test': []}
total_files = sum(len(files) for files in groups.values())

# Сначала считаем, сколько файлов должно быть в каждой выборке
target_counts = {k: int(v * total_files) for k, v in split_ratio.items()}

current_counts = {'train': 0, 'val': 0, 'test': 0}

for key in group_keys:
    files = groups[key]
    # Определяем, куда отправить группу (наименее заполненная в процентах)
    best_split = 'train'
    min_ratio = float('inf')
    
    for split_name in ['train', 'val', 'test']:
        if target_counts[split_name] == 0:
            continue
        ratio = current_counts[split_name] / target_counts[split_name]
        if ratio < min_ratio:
            min_ratio = ratio
            best_split = split_name
            
    splits[best_split].extend(files)
    current_counts[best_split] += len(files)

# Создаем папки и копируем файлы
for split_name, files in splits.items():
    img_out = os.path.join(output_dir, split_name, 'images')
    lbl_out = os.path.join(output_dir, split_name, 'labels')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    
    for f in files:
        shutil.copy2(f['img_src'], os.path.join(img_out, f['img_name']))
        shutil.copy2(f['lbl_src'], os.path.join(lbl_out, f['lbl_name']))

print("\n--- РЕЗУЛЬТАТ ПЕРЕСБОРКИ ---")
print(f"Всего файлов: {total_files}")
for split_name, files in splits.items():
    print(f"{split_name}: {len(files)} файлов ({len(files)/total_files*100:.1f}%)")
print(f"Новый датасет сохранен в папку: {output_dir}")