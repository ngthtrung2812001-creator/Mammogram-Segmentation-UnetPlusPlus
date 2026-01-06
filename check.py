import torch
import sys

# Đường dẫn đến file last_model.pth của bạn
path = "output/GrandMaster_EffB5_GammaV1/last_model.pth" 

try:
    ckpt = torch.load(path, map_location='cpu')
    print(f"Current Epoch: {ckpt.get('epoch')}")
    print(f"Best Dice: {ckpt.get('best_dice_mass')}")
    
    history = ckpt.get('history', {})
    print("\n--- History Lengths ---")
    for k, v in history.items():
        print(f"{k}: {len(v)} items")
        
except Exception as e:
    print(f"Lỗi: {e}")