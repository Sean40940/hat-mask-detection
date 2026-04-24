from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')  # 自動下載預訓練權重

    model.train(
        data=r"E:\USER\Documents\研究生\口罩辨識\帽子辨識\data.yaml",
        epochs=100,
        imgsz=512,
        device=0,
        batch=4,
        workers=2,
        amp=False,
        project=r"E:\USER\Documents\研究生\口罩辨識\帽子辨識\runs",
        name="hat_mask_v1",
        exist_ok=True
    )