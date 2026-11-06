from ultralytics import YOLO

def main():
    # Загружаем ту же базовую модель, что и для E1
    model = YOLO('models/yolov8n.pt')

    model.train(
        data='configs/ppe.yaml',
        epochs=50,
        imgsz=800,               # <--- ИЗМЕНЕННЫЙ ПАРАМЕТР (в E1 было 640)
        batch=8,                 # Уменьшили батч, чтобы не перегрузить оперативку
        seed=42,
        deterministic=True,
        project='runs/ppe',
        name='e2-imgsz800',      # Имя папки для эксперимента
        device='cpu'
    )

if __name__ == '__main__':
    main()