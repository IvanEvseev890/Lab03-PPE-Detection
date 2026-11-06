import os
import cv2
import glob
import shutil  # <-- Добавлен импорт для удаления папок
from ultralytics import YOLO

# Загружаем лучшую модель E2
model = YOLO('runs/detect/runs/ppe/e2-imgsz800/weights/best.pt')
img_dir = 'data_clean/test/images'
lbl_dir = 'data_clean/test/labels'
base_out = 'results/error_galleries'

# --- ОЧИСТКА СТАРЫХ ГАЛЕРЕЙ ---
if os.path.exists(base_out):
    print(f"Очищаем старые галереи в папке: {base_out}...")
    shutil.rmtree(base_out)  # Полностью удаляем старую папку со всем содержимым

# Создаем папки для галерей заново
folders = [
    '1_FP_HighConf', '2_FN_Critical', '3_Low_IoU', 
    '4_Confusion', '5_Small_Occluded', '6_Label_Errors'
]
for f in folders:
    os.makedirs(os.path.join(base_out, f), exist_ok=True)

def load_yolo_labels(lbl_path, w, h):
    boxes = []
    if not os.path.exists(lbl_path): return boxes
    with open(lbl_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                c, x, y, bw, bh = map(float, parts)
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)
                boxes.append({'class': int(c), 'box': [x1, y1, x2, y2]})
    return boxes

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

test_images = glob.glob(f'{img_dir}/*.jpg')
counts = {f: 0 for f in folders}

for img_path in test_images:
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    lbl_path = os.path.join(lbl_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
    gt_boxes = load_yolo_labels(lbl_path, w, h)
    
    # Предсказание модели
    results = model(img_path, verbose=False)[0]
    pred_boxes = []
    for box in results.boxes:
        pred_boxes.append({
            'class': int(box.cls[0]),
            'conf': float(box.conf[0]),
            'box': box.xyxy[0].cpu().numpy().astype(int).tolist()
        })
    
    # 1. FP с высокой уверенностью (> 0.7), которых нет в GT
    for pred in pred_boxes:
        if pred['conf'] > 0.7:
            matched = False
            for gt in gt_boxes:
                if gt['class'] == pred['class'] and calculate_iou(gt['box'], pred['box']) > 0.5:
                    matched = True
                    break
            if not matched:
                cv2.imwrite(f'{base_out}/1_FP_HighConf/fp_{counts["1_FP_HighConf"]}.jpg', img)
                counts['1_FP_HighConf'] += 1
                break

    # 2. Пропуски критичного класса (Hardhat=0, NO-Hardhat=2, Safety Vest=7)
    critical = [0, 2, 7]
    for gt in gt_boxes:
        if gt['class'] in critical:
            matched = False
            for pred in pred_boxes:
                if gt['class'] == pred['class'] and calculate_iou(gt['box'], pred['box']) > 0.5:
                    matched = True
                    break
            if not matched:
                cv2.imwrite(f'{base_out}/2_FN_Critical/fn_{counts["2_FN_Critical"]}.jpg', img)
                counts['2_FN_Critical'] += 1
                break

    # 3. Низкий IoU (0.1 < IoU < 0.5)
    for pred in pred_boxes:
        for gt in gt_boxes:
            if pred['class'] == gt['class']:
                iou = calculate_iou(gt['box'], pred['box'])
                if 0.1 < iou < 0.5:
                    cv2.imwrite(f'{base_out}/3_Low_IoU/low_iou_{counts["3_Low_IoU"]}.jpg', img)
                    counts['3_Low_IoU'] += 1
                    break
        else: continue
        break

    # 4. Путаница (Hardhat <-> NO-Hardhat или Person <-> NO-Safety Vest)
    for pred in pred_boxes:
        for gt in gt_boxes:
            if pred['class'] != gt['class'] and calculate_iou(gt['box'], pred['box']) > 0.5:
                if (pred['class'] in [0, 2] and gt['class'] in [0, 2]) or (pred['class'] in [5, 4] and gt['class'] in [5, 4]):
                    cv2.imwrite(f'{base_out}/4_Confusion/conf_{counts["4_Confusion"]}.jpg', img)
                    counts['4_Confusion'] += 1
                    break
        else: continue
        break

    # 5. Малые или перекрытые объекты (площадь < 32*32)
    for gt in gt_boxes:
        bw = gt['box'][2] - gt['box'][0]
        bh = gt['box'][3] - gt['box'][1]
        if bw * bh < 1024: 
            cv2.imwrite(f'{base_out}/5_Small_Occluded/small_{counts["5_Small_Occluded"]}.jpg', img)
            counts['5_Small_Occluded'] += 1
            break

    # 6. Предположительные ошибки разметки (Label Errors)
    for pred in pred_boxes:
        if pred['conf'] > 0.7:
            same_class_gts = [gt for gt in gt_boxes if gt['class'] == pred['class']]
            max_iou = 0
            for gt in same_class_gts:
                max_iou = max(max_iou, calculate_iou(gt['box'], pred['box']))
            
            if (0.0 < max_iou < 0.3) or (len(same_class_gts) == 0):
                nearby_objects = False
                px, py = (pred['box'][0]+pred['box'][2])/2, (pred['box'][1]+pred['box'][3])/2
                for gt in gt_boxes:
                    gx, gy = (gt['box'][0]+gt['box'][2])/2, (gt['box'][1]+gt['box'][3])/2
                    dist = ((px-gx)**2 + (py-gy)**2)**0.5
                    if dist < 150: 
                        nearby_objects = True
                        break
                
                if nearby_objects:
                    cv2.imwrite(f'{base_out}/6_Label_Errors/label_err_{counts["6_Label_Errors"]}.jpg', img)
                    counts['6_Label_Errors'] += 1
                    break

print("\n--- Сбор галерей ошибок завершен! ---")
for f in folders:
    print(f"{f}: {counts[f]} изображений")
print(f"\nВсе галереи сохранены в папку: {base_out}/")