import os
import shutil
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import BASE_OUTPUT

def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().item()
    elif isinstance(value, list):
        return [tensor_to_float(v) for v in value]
    return value

def export(trainer, save_dir=None):
    if save_dir:
        base_folder = os.path.join(BASE_OUTPUT, save_dir)
    else:
        base_folder = os.path.join(BASE_OUTPUT, "default_run")
        
    os.makedirs(base_folder, exist_ok=True)

    source_files = {
        'last_model.pth': 'last_model.pth',
        'best_dice_mass_model.pth': 'best_dice_mass_model.pth',
        'best_iou_mass_model.pth': 'best_iou_mass_model.pth'
    }

    for src, dst_name in source_files.items():
        if os.path.exists(src):
            dst_path = os.path.join(base_folder, dst_name)
            if os.path.exists(dst_path): os.remove(dst_path)
            shutil.move(src, base_folder)
            print(f"[INFO] Saved model to: {dst_path}")

    # --- XỬ LÝ SỐ LIỆU (FIXED) ---
    checkpoint_path = os.path.join(base_folder, 'last_model.pth')
    if not os.path.exists(checkpoint_path): return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint.get('history', {})
    
    # Lấy số epoch thực tế từ train_loss (vì nó luôn có dữ liệu)
    real_epochs = len(history.get('train_loss', []))
    if real_epochs == 0: return # Không có dữ liệu để vẽ

    epochs = list(range(1, checkpoint.get('epoch', real_epochs) + 1))
    
    # Đảm bảo độ dài epochs khớp với real_epochs
    if len(epochs) > real_epochs: epochs = epochs[:real_epochs]
    
    data = {'epoch': epochs}
    
    keys_to_extract = [
        'train_loss', 'val_loss',
        'train_dice_mass', 'val_dice_mass', 
        'train_iou_mass', 'val_iou_mass',
        'val_dice_norm', 'val_iou_norm' # Norm có thể rỗng
    ]
    
    for k in keys_to_extract:
        values = tensor_to_float(history.get(k, []))
        
        # FIX: Padding bằng NaN nếu thiếu dữ liệu
        if len(values) < real_epochs:
            values = values + [np.nan] * (real_epochs - len(values))
        elif len(values) > real_epochs:
            values = values[:real_epochs]
            
        data[k] = values

    df = pd.DataFrame(data)
    csv_path = os.path.join(base_folder, 'training_history.csv')
    df.to_csv(csv_path, index=False)
    print(f"[INFO] History saved to {csv_path}")

    # --- VẼ BIỂU ĐỒ ---
    try:
        plt.figure(figsize=(18, 6))
        
        # 1. Loss
        plt.subplot(1, 3, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        plt.title('Losses'); plt.legend(); plt.grid(True, alpha=0.3)

        # 2. Dice
        plt.subplot(1, 3, 2)
        plt.plot(df['epoch'], df['train_dice_mass'], label='Train Mass')
        plt.plot(df['epoch'], df['val_dice_mass'], label='Val Mass')
        # Chỉ vẽ Norm nếu có dữ liệu (không toàn NaN)
        if not df['val_dice_norm'].isna().all():
            plt.plot(df['epoch'], df['val_dice_norm'], label='Val Norm', linestyle='--')
        plt.title(f"Dice Score (Best Mass: {checkpoint.get('best_dice_mass', 0):.4f})")
        plt.legend(); plt.grid(True, alpha=0.3)

        # 3. IoU
        plt.subplot(1, 3, 3)
        plt.plot(df['epoch'], df['train_iou_mass'], label='Train Mass')
        plt.plot(df['epoch'], df['val_iou_mass'], label='Val Mass')
        if not df['val_iou_norm'].isna().all():
            plt.plot(df['epoch'], df['val_iou_norm'], label='Val Norm', linestyle='--')
        plt.title(f"IoU Score (Best Mass: {checkpoint.get('best_iou_mass', 0):.4f})")
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(base_folder, "metrics_chart.png")
        plt.savefig(chart_path, dpi=150)
        print(f"[INFO] Chart saved to {chart_path}")
    except Exception as e:
        print(f"[WARN] Error plotting chart: {e}")
    finally:
        plt.close('all')

def export_evaluate(trainer, save_dir=None):
    if save_dir:
        output_folder = os.path.join(BASE_OUTPUT, save_dir)
    else:
        output_folder = BASE_OUTPUT
    os.makedirs(output_folder, exist_ok=True)
    df = pd.DataFrame({
        'ImagePath': trainer.path_list, 'Type': trainer.type_list,
        'Dice': trainer.dice_list, 'IoU': trainer.iou_list
    })
    result_csv = os.path.join(output_folder, "test_metrics_details.csv")
    df.to_csv(result_csv, index=False)