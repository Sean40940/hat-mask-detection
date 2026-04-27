from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt') 

    model.train(
        data=r"C:\Users\user\Desktop\hat.v2i.yolo26\hat_only_merged\data_hat_only.yaml", 
        epochs=100,
        imgsz=512,
        device=0,
        # --- 加入下面這幾行來解決記憶體問題 ---
        batch=4,           # 原本可能是 16，改小一點讓顯卡喘口氣
        workers=2,         # 減少同時讀取圖片的線程
        amp=False,         # 關閉混合精度，預防某些顯卡驅動衝突
        # ------------------------------------
    )