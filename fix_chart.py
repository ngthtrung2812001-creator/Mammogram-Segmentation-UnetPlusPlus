import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- CẤU HÌNH ---
# Đường dẫn file model bị lỗi biểu đồ
CHECKPOINT_PATH = "output/GrandMaster_EffB5_GammaV1/last_model.pth"
SAVE_DIR = "output/GrandMaster_EffB5_GammaV1"

def fix_and_plot():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Không tìm thấy file: {CHECKPOINT_PATH}")
        return

    print(f"🔄 Đang đọc checkpoint: {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    history = checkpoint.get('history', {})
    
    # 1. Xác định số Epoch thực tế (dựa vào train_loss)
    if 'train_loss' not in history or len(history['train_loss']) == 0:
        print("❌ History không có dữ liệu train_loss.")
        return

    real_epochs = len(history['train_loss'])
    epochs = list(range(1, real_epochs + 1))
    print(f"✅ Phát hiện {real_epochs} Epochs dữ liệu.")

    # 2. Chuẩn bị dữ liệu vẽ (Tự động Padding)
    data = {'epoch': epochs}
    keys_to_plot = [
        'train_loss', 'val_loss',
        'train_dice_mass', 'val_dice_mass', 
        'train_iou_mass', 'val_iou_mass'
    ]
    
    # Chỉ lấy các key quan trọng, bỏ qua _norm nếu nó bị rỗng
    for k in keys_to_plot:
        values = history.get(k, [])
        # Chuyển tensor sang float nếu cần
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy().tolist()
            
        # Nếu thiếu dữ liệu -> Điền NaN
        if len(values) < real_epochs:
            print(f"⚠️ Key '{k}' thiếu dữ liệu (Len={len(values)}). Đang điền NaN...")
            values = values + [np.nan] * (real_epochs - len(values))
        # Nếu thừa -> Cắt bớt
        elif len(values) > real_epochs:
            values = values[:real_epochs]
            
        data[k] = values

    df = pd.DataFrame(data)
    
    # Lưu lại CSV đã sửa
    csv_path = os.path.join(SAVE_DIR, 'fixed_history.csv')
    df.to_csv(csv_path, index=False)
    print(f"💾 Đã lưu CSV lịch sử: {csv_path}")

    # 3. Vẽ biểu đồ
    plt.figure(figsize=(18, 6))
    
    # Chart 1: Loss
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Losses')
    plt.legend(); plt.grid(True, alpha=0.3)

    # Chart 2: Dice Mass
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['train_dice_mass'], label='Train Mass')
    plt.plot(df['epoch'], df['val_dice_mass'], label='Val Mass')
    plt.title(f"Dice Mass (Best: {checkpoint.get('best_dice_mass', 0):.4f})")
    plt.legend(); plt.grid(True, alpha=0.3)

    # Chart 3: IoU Mass
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['train_iou_mass'], label='Train Mass')
    plt.plot(df['epoch'], df['val_iou_mass'], label='Val Mass')
    plt.title(f"IoU Mass (Best: {checkpoint.get('best_iou_mass', 0):.4f})")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(SAVE_DIR, "fixed_metrics_chart.png")
    plt.savefig(chart_path, dpi=150)
    print(f"📈 Đã vẽ xong biểu đồ! Kiểm tra tại: {chart_path}")
    plt.close()

if __name__ == "__main__":
    fix_and_plot()