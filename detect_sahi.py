"""
工廠安全偵測系統 - SAHI 增強版 v2

針對俯角監控攝影機優化：
  - 不依賴 COCO 人員偵測（俯角下辨識率極差）
  - 改以「帽子位置」代表工人位置
  - 在帽子附近搜尋口罩，判斷是否合規

使用方式：
  python detect_sahi.py --model best.pt --source factory.avi
  python detect_sahi.py --model best.pt --source factory.avi --output result.mp4
  python detect_sahi.py --model best.pt --source 0   # webcam
"""

import cv2
import torch
import argparse
from pathlib import Path
from datetime import datetime

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── 類別定義（需與訓練時 data.yaml 的 names 順序一致）──────────
HAT_ID  = 0
MASK_ID = 1

# ── 合規狀態顏色與標籤 ─────────────────────────────────────────
STATUS = {
    "ok":      {"color": (0, 200,   0), "label": "OK"},
    "no_mask": {"color": (0, 140, 255), "label": "No Mask!"},
}
# ──────────────────────────────────────────────────────────────


def get_device() -> str:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda:0"
    print("GPU 不可用，使用 CPU")
    return "cpu"


def load_model(hatmask_path: str, device: str, conf: float):
    if not Path(hatmask_path).exists():
        raise FileNotFoundError(f"找不到模型檔案: {hatmask_path}")
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=hatmask_path,
        confidence_threshold=conf,
        device=device,
    )


# ── SAHI 偵測 ──────────────────────────────────────────────────

def sahi_detect(frame, sahi_model, slice_size: int, overlap: float):
    """
    SAHI 切片推理：將影像切成重疊小塊分別推理後合併。
    切片越小，越容易偵測到遠距離的小口罩。
    回傳 hats, masks: [[x1,y1,x2,y2,conf], ...]
    """
    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=0,
    )
    hats, masks = [], []
    for pred in result.object_prediction_list:
        b = pred.bbox
        entry = [int(b.minx), int(b.miny), int(b.maxx), int(b.maxy), pred.score.value]
        if pred.category.id == HAT_ID:
            hats.append(entry)
        elif pred.category.id == MASK_ID:
            masks.append(entry)
    return hats, masks


# ── 合規判斷（以帽子為人員位置基準）──────────────────────────────

def has_nearby_mask(hat_box: list, masks: list, search_ratio: float = 2.0) -> bool:
    """
    在帽子周圍搜尋口罩。
    search_ratio: 搜尋半徑 = 帽子寬度 * search_ratio
    俯角攝影機下口罩通常在帽子框的正下方或旁邊。
    """
    hx = (hat_box[0] + hat_box[2]) / 2
    hy = (hat_box[1] + hat_box[3]) / 2
    hat_w = hat_box[2] - hat_box[0]
    hat_h = hat_box[3] - hat_box[1]
    search_r = max(hat_w, hat_h) * search_ratio

    for mask in masks:
        mx = (mask[0] + mask[2]) / 2
        my = (mask[1] + mask[3]) / 2
        dist = ((mx - hx) ** 2 + (my - hy) ** 2) ** 0.5
        if dist < search_r:
            return True
    return False


# ── 繪圖 ───────────────────────────────────────────────────────

def draw_frame(frame, hats: list, masks: list, search_ratio: float) -> tuple:
    """
    繪製偵測結果：
      - 每個帽子 = 一個工人，檢查附近是否有口罩
      - 綠框 = OK（帽子+口罩都有）
      - 橘框 = No Mask!（有帽子但找不到口罩）
      - 青色小框 = 偵測到的口罩位置
    """
    # 先畫口罩框（青色，細框）
    for x1, y1, x2, y2, conf in masks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 1)
        cv2.putText(frame, f"mask {conf:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)

    violations = 0
    for hat in hats:
        x1, y1, x2, y2, conf = hat
        compliant = has_nearby_mask(hat, masks, search_ratio)
        status = "ok" if compliant else "no_mask"
        color  = STATUS[status]["color"]
        label  = STATUS[status]["label"]
        if not compliant:
            violations += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # HUD
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    workers = len(hats)
    cv2.putText(frame, f"Workers: {workers}  Violations: {violations}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, ts, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    return frame, violations


# ── 主迴圈 ─────────────────────────────────────────────────────

def run(args):
    device = get_device()
    sahi_model = load_model(args.model, device, args.conf)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"無法開啟來源: {source}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"輸出影片: {args.output}")

    frame_no, total_violations = 0, 0
    print("開始偵測 (按 Q 鍵離開)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        hats, masks  = sahi_detect(frame, sahi_model, args.slice_size, args.overlap)
        frame, viols = draw_frame(frame, hats, masks, args.search_ratio)
        total_violations += viols

        if writer:
            writer.write(frame)

        cv2.imshow("Factory Safety Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"完成，共處理 {frame_no} 幀，累計違規 {total_violations} 次")


# ── CLI 參數 ───────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="工廠安全偵測系統 (SAHI 增強版 v2 - 俯角優化)")
    p.add_argument(
        "--model", default="best.pt",
        help="帽子/口罩模型路徑 (.pt)，預設: best.pt",
    )
    p.add_argument(
        "--source", default="0",
        help="影片路徑或攝影機編號，預設: 0 (webcam)",
    )
    p.add_argument(
        "--output", default=None,
        help="儲存輸出影片路徑 (例如: result.mp4)",
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="偵測信心門檻，預設: 0.25（較低以提高口罩召回率）",
    )
    p.add_argument(
        "--slice-size", type=int, default=320,
        help="SAHI 切片大小，預設: 320（較小切片讓遠距口罩更清晰）",
    )
    p.add_argument(
        "--overlap", type=float, default=0.3,
        help="SAHI 切片重疊比例，預設: 0.3",
    )
    p.add_argument(
        "--search-ratio", type=float, default=2.0,
        help="在帽子周圍幾倍帽寬內搜尋口罩，預設: 2.0",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
