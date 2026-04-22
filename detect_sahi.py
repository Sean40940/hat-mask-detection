"""
工廠安全偵測系統 - SAHI 增強版

SAHI 原理：把整張影像切成多個重疊小塊 → 各自推理 → 合併結果
效果：大幅改善遠距離、小尺寸口罩的偵測率

使用方式：
  python detect_sahi.py --model best.pt --source factory.avi
  python detect_sahi.py --model best.pt --source 0           # webcam
  python detect_sahi.py --model best.pt --source 0 --output result.mp4
"""

import cv2
import torch
import argparse
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ── 類別定義（需與訓練時 data.yaml 的 names 順序一致）──────────
HAT_ID  = 0
MASK_ID = 1

# 人員偵測用 COCO 模型中 person 的 class id
COCO_PERSON_ID = 0

# ── 合規狀態顏色與標籤 ─────────────────────────────────────────
STATUS = {
    "ok":      {"color": (0, 200,   0), "label": "OK"},
    "no_mask": {"color": (0, 140, 255), "label": "No Mask!"},
    "no_hat":  {"color": (0, 140, 255), "label": "No Hat!"},
    "no_ppe":  {"color": (0,   0, 220), "label": "No PPE!"},
}
# ──────────────────────────────────────────────────────────────


def get_device() -> str:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda:0"
    print("GPU 不可用，使用 CPU")
    return "cpu"


def load_models(hatmask_path: str, person_path: str | None, device: str, conf: float):
    """載入帽子/口罩 SAHI 模型與人員偵測模型。"""
    if not Path(hatmask_path).exists():
        raise FileNotFoundError(f"找不到模型檔案: {hatmask_path}")

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=hatmask_path,
        confidence_threshold=conf,
        device=device,
    )

    person_model = None
    if person_path and Path(person_path).exists():
        person_model = YOLO(person_path)
        print(f"人員偵測模型載入: {person_path}")
    elif person_path:
        # ultralytics 會自動下載
        person_model = YOLO(person_path)
        print(f"人員偵測模型載入: {person_path}")

    return sahi_model, person_model


# ── 偵測函式 ───────────────────────────────────────────────────

def detect_persons(frame, model, device: str) -> list[list[int]]:
    """回傳人員 bounding boxes: [[x1,y1,x2,y2], ...]"""
    if model is None:
        return []
    results = model.predict(
        frame,
        device=device,
        conf=0.4,
        classes=[COCO_PERSON_ID],
        verbose=False,
    )
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append([x1, y1, x2, y2])
    return boxes


def sahi_detect(frame, sahi_model, slice_size: int, overlap: float):
    """
    SAHI 切片推理：
      1. 將整張影像切成 slice_size x slice_size 的重疊小塊
      2. 每塊分別推理
      3. 合併所有結果並做 NMS 去重複
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


# ── 合規判斷 ───────────────────────────────────────────────────

def center_inside(item_box: list, person_box: list) -> bool:
    """判斷 item_box 的中心點是否在 person_box 內。"""
    cx = (item_box[0] + item_box[2]) / 2
    cy = (item_box[1] + item_box[3]) / 2
    return (person_box[0] <= cx <= person_box[2] and
            person_box[1] <= cy <= person_box[3])


def classify_worker(person_box: list, hats: list, masks: list) -> str:
    """回傳合規狀態: 'ok' / 'no_mask' / 'no_hat' / 'no_ppe'"""
    has_hat  = any(center_inside(h, person_box) for h in hats)
    has_mask = any(center_inside(m, person_box) for m in masks)
    if has_hat and has_mask:
        return "ok"
    if has_hat:
        return "no_mask"
    if has_mask:
        return "no_hat"
    return "no_ppe"


# ── 繪圖 ───────────────────────────────────────────────────────

def draw_frame(frame, persons: list, hats: list, masks: list) -> tuple:
    """繪製帽子/口罩框、人員合規狀態框與 HUD。回傳 (frame, violation_count)。"""
    # 帽子框（橙色）
    for x1, y1, x2, y2, conf in hats:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, f"hat {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

    # 口罩框（青色）
    for x1, y1, x2, y2, conf in masks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
        cv2.putText(frame, f"mask {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)

    violations = 0
    for person_box in persons:
        status = classify_worker(person_box, hats, masks)
        color  = STATUS[status]["color"]
        label  = STATUS[status]["label"]
        if status != "ok":
            violations += 1
        x1, y1, x2, y2 = person_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # 右上角 HUD
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Workers: {len(persons)}  Violations: {violations}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, ts, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    return frame, violations


# ── 主迴圈 ─────────────────────────────────────────────────────

def run(args):
    device = get_device()
    sahi_model, person_model = load_models(
        args.model, args.person_model, device, args.conf
    )

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

        persons     = detect_persons(frame, person_model, device)
        hats, masks = sahi_detect(frame, sahi_model, args.slice_size, args.overlap)
        frame, viols = draw_frame(frame, persons, hats, masks)
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
    p = argparse.ArgumentParser(description="工廠安全偵測系統 (SAHI 增強版)")
    p.add_argument(
        "--model", default="best.pt",
        help="帽子/口罩模型路徑 (.pt)，預設: best.pt",
    )
    p.add_argument(
        "--person-model", default="yolo11n.pt",
        help="人員偵測模型 (COCO 權重)，留空則停用人員偵測，預設: yolo11n.pt",
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
        "--conf", type=float, default=0.35,
        help="帽子/口罩偵測信心門檻，預設: 0.35",
    )
    p.add_argument(
        "--slice-size", type=int, default=512,
        help="SAHI 切片大小 (像素)，預設: 512",
    )
    p.add_argument(
        "--overlap", type=float, default=0.2,
        help="SAHI 切片重疊比例 0~1，預設: 0.2",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
