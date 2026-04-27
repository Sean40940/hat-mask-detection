from ultralytics import YOLO
import os
import torch

def run_prediction():
    # 1. 路徑設定 (請確認資料夾路徑正確)
    model_path = r"E:\USER\Documents\研究生\口罩辨識\帽子辨識\runs\hat_mask_v1\weights\best.pt"
    video_path = r"E:\USER\Documents\研究生\口罩辨識\帽子辨識\runs\detect\predict_optimized2\factory01.avi"
    output_project = r'E:\USER\Documents\研究生\口罩辨識\帽子辨識\runs\detect'

    # 檢查檔案是否存在
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案：{model_path}")
        return
    if not os.path.exists(video_path):
        print(f"❌ 找不到影片檔案：{video_path}")
        return

    # 2. 載入模型
    print("正在載入模型...")
    model = YOLO(model_path)

    # 檢查 GPU 狀態
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用設備: {torch.cuda.get_device_name(0) if device==0 else 'CPU'}")

    # 3. 執行預測與參數優化
    print("開始執行辨識，請稍候...")
    results = model.predict(
        source=video_path,
        project=output_project,
        name='predict_optimized', # 儲存資料夾名稱
        save=True,               # 儲存標記後的影片
        device=device,           # 自動選擇 GPU
        
        # --- 核心優化參數 ---
        conf=0.4,                # [重要] 提高門檻，過濾掉那些 < 0.3 的誤判框 (如桌子、麵團)
        iou=0.45,                # [重要] 降低重疊容忍度，解決「一人兩框」的重複問題
        agnostic_nms=True,       # [進階] 不分標籤進行去重疊，防止帽子跟口罩框疊在一起打架
        
        # --- 其他輔助設定 ---
        line_width=2,            # 讓標記框的線條細一點，比較好觀察重疊處
        show_conf=True,          # 顯示信心值，方便你觀察後續是否要調整 conf
        show_labels=True         # 顯示類別標籤
    )

    print(f"✅ 預測完成！")
    print(f"📁 結果儲存於: {os.path.join(output_project, 'predict_optimized')}")

if __name__ == "__main__":
    run_prediction()