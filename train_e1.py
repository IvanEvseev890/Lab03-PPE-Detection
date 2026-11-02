from ultralytics import YOLO

def main():
    # Загружаем предобученную модель (базовую)
    model = YOLO('models/yolov8n.pt')

    # Запускаем обучение по параметрам из лабораторной
    model.train(
        data='configs/ppe.yaml', # Наш новый конфиг с чистым датасетом
        epochs=50,               # 50 эпох
        imgsz=640,               # Размер изображения
        batch=16,                # Размер пакета (если не хватит оперативки, поставь 8)
        seed=42,                 # Фиксация генератора
        deterministic=True,      # Детерминированность
        project='runs/ppe',      # Папка для сохранения результатов
        name='e1-opora',         # Имя эксперимента
        device='cpu'             # Явно указываем CPU (так как ставили CPU версию)
    )

    print("\n--- Обучение E1 завершено! ---")
    print("Веса сохранены в папке: runs/ppe/e1-opora/weights/best.pt")

if __name__ == '__main__':
    main()