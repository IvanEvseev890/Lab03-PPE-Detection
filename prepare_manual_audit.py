import os
import glob
import random
import shutil

# Пути
img_dir = 'data_clean/train/images'  # Берем из трейна, чтобы не тратить тест
lbl_dir = 'data_clean/train/labels'
out_dir = 'results/manual_audit'

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(os.path.join(out_dir, '30_random'), exist_ok=True)
os.makedirs(os.path.join(out_dir, '10_occluded'), exist_ok=True)

images = glob.glob(f'{img_dir}/*.jpg')
random.seed(42) # Фиксируем сид
random.shuffle(images)

# 1. Копируем 30 случайных
for img in images[:30]:
    shutil.copy(img, os.path.join(out_dir, '30_random', os.path.basename(img)))

# 2. Ищем 10 с перекрытием (маленькие объекты)
occluded_found = 0
for img_path in images:
    if occluded_found >= 10: break
    
    lbl_path = os.path.join(lbl_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
    if not os.path.exists(lbl_path): continue
    
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                _, _, _, bw, bh = map(float, parts)
                # Если площадь объекта меньше 32x32 пикселей (0.05 от 640)
                if (bw * 640) * (bh * 640) < 1024: 
                    shutil.copy(img_path, os.path.join(out_dir, '10_occluded', os.path.basename(img_path)))
                    occluded_found += 1
                    break # Берем одну картинку только один раз
                    
print(f"Готово! Картинки лежат в {out_dir}")
print(f"Случайных: 30, Перекрытых: {occluded_found}")