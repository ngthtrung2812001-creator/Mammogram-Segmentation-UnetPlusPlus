import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import os
import time
import gc

# Chỉ import các hàm cần thiết từ utils
from utils import dice_coeff_hard, iou_core_hard, visualize_prediction

class Trainer:
    def __init__(self, model, optimizer, criterion, num_epochs=50, patience=20, device=None):
        """
        Trainer độc lập, không phụ thuộc vào file config toàn cục.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Tracking metrics
        self.early_stop_counter = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice_mass': [], 'val_dice_mass': [],
            'train_dice_norm': [], 'val_dice_norm': [],
            'train_iou_mass': [],  'val_iou_mass': [],
            'train_iou_norm': [],  'val_iou_norm': []
        }
        
        # Best metrics
        self.best_dice_mass = 0.0
        self.best_iou_mass = 0.0
        self.best_val_loss = float('inf')
        
        self.best_epoch_dice = 0
        self.best_epoch_iou = 0
        self.best_epoch_loss = 0
        
        self.log_interval = 1
        self.start_epoch = 0

        # AMP (Automatic Mixed Precision)
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Scheduler (Có thể đưa ra ngoài main nếu muốn tùy chỉnh cao hơn)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),   
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history,
            'best_dice_mass': self.best_dice_mass,
            'best_iou_mass': self.best_iou_mass,
            'best_val_loss': self.best_val_loss,
            'best_epoch_dice': self.best_epoch_dice,
            'best_epoch_iou': self.best_epoch_iou,
            'best_epoch_loss': self.best_epoch_loss
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, path):
        print(f"[INFO] Loading checkpoint: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            self.start_epoch = checkpoint.get('epoch', 0)
            self.history = checkpoint.get('history', self.history)
            
            self.best_dice_mass = checkpoint.get('best_dice_mass', 0.0)
            self.best_iou_mass = checkpoint.get('best_iou_mass', 0.0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            self.best_epoch_dice = checkpoint.get('best_epoch_dice', 0)
            self.best_epoch_iou = checkpoint.get('best_epoch_iou', 0)
            self.best_epoch_loss = checkpoint.get('best_epoch_loss', 0)
            
            print(f"[INFO] Loaded checkpoint from epoch {self.start_epoch}")
        except Exception as e:
            print(f"[ERROR] Could not load checkpoint: {e}")

    def run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        
        epoch_loss = 0.0
        
        # Accumulators
        total_dice_mass, total_iou_mass = 0.0, 0.0
        count_mass = 0
        
        total_dice_norm, total_iou_norm = 0.0, 0.0
        count_norm = 0
        
        desc = "Training" if is_train else "Validation"
        loader_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False)
        
        for i, (images, masks, _) in loader_bar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Tính Metrics (No Grad)
                with torch.no_grad():
                    batch_dice = dice_coeff_hard(outputs, masks) # Tensor [B]
                    batch_iou = iou_core_hard(outputs, masks)    # Tensor [B]
                    
                    # Phân loại Mass / Normal
                    masks_flat = masks.view(masks.size(0), -1)
                    mask_sums = masks_flat.sum(dim=1)
                    
                    is_mass = (mask_sums > 0)
                    is_norm = (mask_sums == 0)
                    
                    # Cộng dồn
                    if is_mass.any():
                        total_dice_mass += batch_dice[is_mass].sum().item()
                        total_iou_mass  += batch_iou[is_mass].sum().item()
                        count_mass += is_mass.sum().item()
                    
                    if is_norm.any():
                        total_dice_norm += batch_dice[is_norm].sum().item()
                        total_iou_norm  += batch_iou[is_norm].sum().item()
                        count_norm += is_norm.sum().item()

            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update scheduler per step (cho CosineAnnealingWarmRestarts)
                self.scheduler.step(self.start_epoch + i / len(loader)) 

            epoch_loss += loss.item()
            
            # Progress bar update
            curr_d_mass = total_dice_mass / count_mass if count_mass > 0 else 0.0
            curr_d_norm = total_dice_norm / count_norm if count_norm > 0 else 0.0
            
            if (i + 1) % self.log_interval == 0:
                loader_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}", 
                    'D_Mass': f"{curr_d_mass:.3f}", 
                    'D_Norm': f"{curr_d_norm:.3f}"
                })
        
        # Tổng kết epoch
        avg_loss = epoch_loss / len(loader)
        final_dice_mass = total_dice_mass / count_mass if count_mass > 0 else 0.0
        final_iou_mass  = total_iou_mass / count_mass if count_mass > 0 else 0.0
        final_dice_norm = total_dice_norm / count_norm if count_norm > 0 else 0.0
        final_iou_norm  = total_iou_norm / count_norm if count_norm > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'dice_mass': final_dice_mass, 'iou_mass': final_iou_mass,
            'dice_norm': final_dice_norm, 'iou_norm': final_iou_norm
        }

    def train(self, train_loader, val_loader, resume_path=None):
        print("=" * 40)
        print(f"🚀 DEVICE: {self.device}")
        print(f"🔄 EPOCHS: {self.num_epochs}")
        print(f"🛑 PATIENCE: {self.patience}")
        print("=" * 40)

        if resume_path:
            self.load_checkpoint(resume_path)
        
        start_time = time.time()

        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch 
            
            # 1. Training
            train_res = self.run_epoch(train_loader, is_train=True)
            
            # 2. Validation
            with torch.no_grad():
                val_res = self.run_epoch(val_loader, is_train=False)

            # 3. Logging & History
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"\n[Epoch {epoch+1}/{self.num_epochs}] LR: {current_lr:.2e}")
            print(f"   Train Loss: {train_res['loss']:.4f} | Mass Dice: {train_res['dice_mass']:.3f} | Norm Dice: {train_res['dice_norm']:.3f}")
            print(f"   Val Loss:   {val_res['loss']:.4f} | Mass Dice: {val_res['dice_mass']:.3f} | Norm Dice: {val_res['dice_norm']:.3f}")

            self.history['train_loss'].append(train_res['loss'])
            self.history['val_loss'].append(val_res['loss'])
            self.history['train_dice_mass'].append(train_res['dice_mass'])
            self.history['val_dice_mass'].append(val_res['dice_mass'])
            self.history['train_iou_mass'].append(train_res['iou_mass'])
            self.history['val_iou_mass'].append(val_res['iou_mass'])
            
            # 4. Checkpointing
            # Luôn lưu bản mới nhất
            self.save_checkpoint(epoch + 1, 'last_model.pth')
            
            # Lưu Best Dice
            if val_res['dice_mass'] > self.best_dice_mass:
                self.best_dice_mass = val_res['dice_mass']
                self.best_epoch_dice = epoch + 1
                self.save_checkpoint(epoch + 1, 'best_dice_mass_model.pth')
                print(f"   🔥 New Best Dice Mass: {self.best_dice_mass:.4f}")

            # Lưu Best IoU
            if val_res['iou_mass'] > self.best_iou_mass:
                self.best_iou_mass = val_res['iou_mass']
                self.best_epoch_iou = epoch + 1
                self.save_checkpoint(epoch + 1, 'best_iou_mass_model.pth')
                print(f"   🔥 New Best IoU Mass: {self.best_iou_mass:.4f}")

            # 5. Early Stopping (dựa trên Loss)
            if val_res['loss'] < self.best_val_loss:
                self.best_val_loss = val_res['loss']
                self.best_epoch_loss = epoch + 1
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                print(f"   ⏳ EarlyStopping counter: {self.early_stop_counter}/{self.patience}")

            if self.early_stop_counter >= self.patience:
                print(f"\n[STOP] Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Cleanup RAM/VRAM
            gc.collect()
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"\n✅ Training Finished in {total_time // 60:.0f}m {total_time % 60:.0f}s")

    def evaluate(self, test_loader, checkpoint_path=None, save_visuals=False, output_dir="test_results"):
        """
        Đánh giá model trên tập test và tùy chọn lưu ảnh kết quả.
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Reset lists để export CSV sau này
        self.dice_list, self.iou_list, self.path_list, self.type_list = [], [], [], []
        
        total_dice_mass, count_mass = 0.0, 0
        total_dice_norm, count_norm = 0.0, 0
        
        if save_visuals:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Visualizations will be saved to: {output_dir}")

        with torch.no_grad():
            test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating")
            
            for i, (images, masks, image_paths) in test_bar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Inference
                logits = self.model(images)
                
                # Tính metrics cứng (cho báo cáo)
                batch_dices = dice_coeff_hard(logits, masks)
                batch_ious = iou_core_hard(logits, masks)
                
                # Chuẩn bị dữ liệu vẽ (nếu cần)
                if save_visuals:
                    probs = torch.sigmoid(logits)
                    #preds = (probs > 0.5).float() # Thresholding 
                    # Lưu ý: visualize_prediction trong utils mới đã tự xử lý sigmoid, nên ta truyền logits hoặc probs đều được,
                    # nhưng truyền probs/logits thì hàm visualize sẽ linh hoạt hơn. 
                    # Ở đây tôi truyền raw logits để utils tự xử lý (như logic utils mới tôi gửi).
                
                # Lặp từng ảnh trong batch
                for j in range(images.size(0)):
                    d = batch_dices[j].item()
                    ious = batch_ious[j].item()
                    path = image_paths[j]
                    
                    # Phân loại
                    is_normal = (masks[j].sum() == 0)
                    current_type = "Normal" if is_normal else "Mass"
                    
                    # Lưu thông tin
                    self.dice_list.append(d)
                    self.iou_list.append(ious)
                    self.path_list.append(path)
                    self.type_list.append(current_type)
                    
                    # Cộng dồn
                    if is_normal:
                        total_dice_norm += d
                        count_norm += 1
                    else:
                        total_dice_mass += d
                        count_mass += 1
                        
                    # Vẽ và lưu ảnh (Chỉ lưu Mass hoặc lưu cả 2 tùy bạn, code này lưu hết)
                    if save_visuals:
                        file_name = os.path.basename(path)
                        prefix = "NORM" if is_normal else "MASS"
                        save_name = f"pred_{prefix}_D{d:.2f}_{file_name}"
                        save_full_path = os.path.join(output_dir, save_name)
                        
                        visualize_prediction(
                            img_tensor=images[j],
                            mask_tensor=masks[j],
                            pred_tensor=logits[j], # Truyền logits để utils tự sigmoid
                            save_path=save_full_path,
                            iou_score=ious,
                            dice_score=d
                        )

        # Tổng hợp kết quả
        avg_dice_mass = total_dice_mass / count_mass if count_mass > 0 else 0.0
        avg_dice_norm = total_dice_norm / count_norm if count_norm > 0 else 0.0
        
        print(f"\n{'='*40}")
        print(f"📊 EVALUATION REPORT")
        print(f"{'='*40}")
        print(f"🩺 Mass Samples ({count_mass}): Avg Dice = {avg_dice_mass:.4f}")
        print(f"🍀 Norm Samples ({count_norm}): Avg Dice = {avg_dice_norm:.4f}")
        print(f"{'='*40}\n")
        
        return avg_dice_mass, avg_dice_norm