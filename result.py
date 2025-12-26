import os
import shutil
import torch
import pandas as pd
import numpy as np

# --- FIX LỖI PLT TRÊN LINUX SERVER ---
import matplotlib
matplotlib.use('Agg') # Quan trọng: Phải đặt trước khi import pyplot
import matplotlib.pyplot as plt

from config import BASE_OUTPUT

def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().item()
    elif isinstance(value, list):
        return [tensor_to_float(v) for v in value]
    return value

def export(trainer, save_dir=None):
    # Xác định thư mục output
    if save_dir:
        base_folder = os.path.join(BASE_OUTPUT, save_dir)
    else:
        # Nếu không có tên saveas, dùng timestamp hoặc tên mặc định
        base_folder = os.path.join(BASE_OUTPUT, "default_run")
        
    os.makedirs(base_folder, exist_ok=True)

    source_files = {
        'last_model.pth': 'last_model.pth',
        'best_dice_mass_model.pth': 'best_dice_mass_model.pth',
        'best_iou_mass_model.pth': 'best_iou_mass_model.pth'
    }

    # Di chuyển các file model đã lưu (thường trainer lưu ở thư mục gốc chạy lệnh)
    for src, dst_name in source_files.items():
        if os.path.exists(src):
            dst_path = os.path.join(base_folder, dst_name)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.move(src, base_folder)
            print(f"[INFO] Saved model to: {dst_path}")

    # --- XỬ LÝ SỐ LIỆU (HISTORY) ---
    checkpoint_path = os.path.join(base_folder, 'last_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Không tìm thấy checkpoint để xuất lịch sử tại {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint.get('history', {})
    
    # Trích xuất dữ liệu an toàn
    data = {'epoch': list(range(1, checkpoint.get('epoch', 0) + 1))}
    
    keys_to_extract = [
        'train_loss', 'val_loss',
        'train_dice_mass', 'val_dice_mass', 
        'train_iou_mass', 'val_iou_mass',
        'train_dice_norm', 'val_dice_norm',
        'train_iou_norm', 'val_iou_norm'
    ]
    
    for k in keys_to_extract:
        values = tensor_to_float(history.get(k, []))
        # Cắt hoặc độn cho bằng số epoch (đề phòng lỗi)
        data[k] = values[:len(data['epoch'])]

    # Lưu CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(base_folder, 'training_history.csv')
    df.to_csv(csv_path, index=False)
    print(f"[INFO] History saved to {csv_path}")

    # --- VẼ BIỂU ĐỒ (Dùng backend Agg an toàn) ---
    try:
        plt.figure(figsize=(18, 6))
        
        # 1. Loss Chart
        plt.subplot(1, 3, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.title('Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Dice Chart
        plt.subplot(1, 3, 2)
        plt.plot(df['epoch'], df['train_dice_mass'], label='Train Mass')
        plt.plot(df['epoch'], df['val_dice_mass'], label='Val Mass')
        plt.plot(df['epoch'], df['val_dice_norm'], label='Val Norm', linestyle='--')
        plt.title(f"Dice Score (Best Mass: {checkpoint.get('best_dice_mass', 0):.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. IoU Chart
        plt.subplot(1, 3, 3)
        plt.plot(df['epoch'], df['train_iou_mass'], label='Train Mass')
        plt.plot(df['epoch'], df['val_iou_mass'], label='Val Mass')
        plt.plot(df['epoch'], df['val_iou_norm'], label='Val Norm', linestyle='--')
        plt.title(f"IoU Score (Best Mass: {checkpoint.get('best_iou_mass', 0):.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(base_folder, "metrics_chart.png")
        plt.savefig(chart_path, dpi=150)
        print(f"[INFO] Chart saved to {chart_path}")
    except Exception as e:
        print(f"[WARN] Error plotting chart: {e}")
    finally:
        plt.close('all') # Đóng figure để giải phóng RAM

def export_evaluate(trainer, save_dir=None):
    if save_dir:
        output_folder = os.path.join(BASE_OUTPUT, save_dir)
    else:
        output_folder = BASE_OUTPUT
        
    os.makedirs(output_folder, exist_ok=True)
    
    df = pd.DataFrame({
        'ImagePath': trainer.path_list,
        'Type': trainer.type_list,
        'Dice': trainer.dice_list,
        'IoU': trainer.iou_list
    })
    
    result_csv = os.path.join(output_folder, "test_metrics_details.csv")
    df.to_csv(result_csv, index=False)
    print(f"[INFO] Evaluation details saved to {result_csv}")