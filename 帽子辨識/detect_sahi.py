"""
帽子/口罩偵測系統 - SAHI 增強版

使用方式：
  python detect_sahi.py
  python detect_sahi.py --source factory.avi --output result.mp4
  python detect_sahi.py --conf 0.2 --slice-size 256
"""

import cv2
import torch
import argparse
from pathlib import Path
from datetime import datetime

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

HAT_ID  = 0
MASK_ID = 1

HAT_COLOR  = (255, 165,   0)  # 橙色
MASK_COLOR = (  0, 220, 220)  # 青色


def get_device() -> str:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda:0"
    print("GPU 不可用，使用 CPU")
    return "cpu"


def load_model(model_path: str, device: str, conf: float):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"找不到模型: {model_path}")
    print(f"載入模型: {model_path}")
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )


def sahi_detect(frame, model, slice_size: int, overlap: float):
    result = get_sliced_prediction(
        frame, model,
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


def draw_frame(frame, hats: list, masks: list) -> None:
    for x1, y1, x2, y2, conf in hats:
        cv2.rectangle(frame, (x1, y1), (x2, y2), HAT_COLOR, 2)
        cv2.putText(frame, f"hat {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HAT_COLOR, 1)

    for x1, y1, x2, y2, conf in masks:
        cv2.rectangle(frame, (x1, y1), (x2, y2), MASK_COLOR, 2)
        cv2.putText(frame, f"mask {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MASK_COLOR, 1)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Hat: {len(hats)}  Mask: {len(masks)}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, ts, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


def run(args):
    device = get_device()
    model  = load_model(args.model, device, args.conf)

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

    print("開始偵測 (按 Q 鍵離開)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hats, masks = sahi_detect(frame, model, args.slice_size, args.overlap)
        draw_frame(frame, hats, masks)

        if writer:
            writer.write(frame)

        cv2.imshow("Hat & Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("完成")


def parse_args():
    p = argparse.ArgumentParser(description="帽子/口罩偵測 (SAHI 增強版)")
    p.add_argument(
        "--model",
        default=r"e:\USER\Documents\研究生\口罩辨識\runs\detect\train\weights\best.pt",
    )
    p.add_argument(
        "--source",
        default=r"E:\USER\Documents\研究生\口罩辨識\帽子辨識\Video\t2.mp4",
    )
    p.add_argument("--output",     default=None)
    p.add_argument("--conf",       type=float, default=0.5)
    p.add_argument("--slice-size", type=int,   default=512)
    p.add_argument("--overlap",    type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
