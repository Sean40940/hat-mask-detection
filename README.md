# 工廠帽子／口罩偵測系統

基於 YOLOv11 + SAHI 的工廠安全防護裝備（帽子、口罩）偵測系統，針對俯角監視器畫面設計，支援即時影片串流與離線影片分析。

## 功能特色

- 使用 **SAHI（Slicing Aided Hyper Inference）** 切片推理，改善遠距離小物件的偵測率
- 支援 GPU 加速（自動偵測 CUDA）
- 可輸出標注後的影片結果
- 針對工廠俯角場景的資料增強訓練策略

## 偵測類別

| 類別 | 顏色 |
|------|------|
| hat（帽子） | 橙色 |
| mask（口罩） | 青色 |

## 環境需求

- Python 3.10+
- CUDA（選用，建議使用以加速推理）

## 安裝

```bash
pip install ultralytics sahi opencv-python torch
```

## 專案結構

```
口罩辨識/
├── hat-mask-detection/
│   ├── hat_new.py          # 訓練腳本
│   └── data.yaml           # 資料集設定
├── 帽子辨識/
│   ├── detect_sahi.py      # 偵測主程式（SAHI 增強版）
│   ├── train/              # 訓練集
│   ├── valid/              # 驗證集
│   └── Video/              # 測試影片
└── runs/
    └── detect/train/
        └── weights/
            ├── best.pt     # 最佳模型權重
            └── last.pt     # 最後一個 epoch 權重
```

## 訓練

```bash
python hat-mask-detection/hat_new.py
```

### 訓練資料集格式（data.yaml）

```yaml
train: <訓練集圖片路徑>
val: <驗證集圖片路徑>
nc: 2
names: ['hat', 'mask']
```

### 訓練參數說明

| 參數 | 值 | 說明 |
|------|----|------|
| epochs | 100 | 訓練回合數 |
| imgsz | 640 | 輸入影像大小 |
| batch | 8 | 批次大小 |
| degrees | 30.0 | 旋轉增強（模擬俯角） |
| mosaic | 1.0 | Mosaic 增強（密集場景） |
| copy_paste | 0.3 | 物件複製貼上（模擬遮擋） |
| scale | 0.6 | 縮放增強（小物件） |

## 推理偵測

```bash
# 使用預設設定（t2.mp4）
python 帽子辨識/detect_sahi.py

# 指定影片來源
python 帽子辨識/detect_sahi.py --source path/to/video.mp4

# 指定模型與信心門檻
python 帽子辨識/detect_sahi.py --model runs/detect/train/weights/best.pt --conf 0.5

# 儲存輸出影片
python 帽子辨識/detect_sahi.py --source video.mp4 --output result.mp4
```

### 推理參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | best.pt 路徑 | 模型權重路徑 |
| `--source` | t2.mp4 路徑 | 影片路徑或攝影機編號（0, 1, ...） |
| `--output` | 無 | 輸出影片儲存路徑 |
| `--conf` | 0.5 | 信心門檻（越高越嚴格，誤報越少） |
| `--slice-size` | 512 | SAHI 切片大小（像素），越小偵測越細但越慢 |
| `--overlap` | 0.2 | 切片重疊比例（0~1） |

## 訓練結果

| 指標 | 數值 |
|------|------|
| mAP50 | ~0.958 |
| Precision | ~0.958 |
| Recall | ~0.889 |
| hat recall | 0.98 |
| mask recall | 0.87 |

## 已知限制

- 口罩訓練資料（385 張）遠少於帽子（2961 張），口罩偵測效果相對較弱
- 建議補充口罩資料至 1500 張以上後重新訓練
