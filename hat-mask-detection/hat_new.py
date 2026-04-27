from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'e:\USER\Documents\研究生\口罩辨識\帽子辨識\runs\hat_mask_v1\weights\best.pt')

    yaml_path = r'e:\USER\Documents\研究生\口罩辨識\hat-mask-detection\data.yaml'

    model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=8,

    # ── 針對工廠燈光變化 ──
    hsv_h=0.015,        # 色相微調
    hsv_s=0.5,          # 飽和度（燈光顏色差異）
    hsv_v=0.5,          # 明度（陰暗區域、背光）

    # ── 針對俯角場景 ──
    degrees=30.0,       # 旋轉角度（俯角各方向都有可能）
    fliplr=0.5,         # 水平翻轉
    flipud=0.3,         # 垂直翻轉（俯角特別有用）

    # ── 針對人員遮擋、密集場景 ──
    mosaic=1.0,         # mosaic 增強（預設就開，確保是 1.0）
    copy_paste=0.3,     # 隨機複製貼上物件（模擬密集場景）
    erasing=0.3,        # 隨機遮擋（模擬人員重疊）

    # ── 針對遠距工人（物件偏小） ──
    scale=0.6,          # 縮放範圍（讓模型學到更小的物件）
    translate=0.1,      # 位移
)