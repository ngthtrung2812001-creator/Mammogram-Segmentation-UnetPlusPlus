import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# --- FIX LỖI PLT TRÊN LINUX SERVER ---
import matplotlib
matplotlib.use('Agg') # Quan trọng: Bắt buộc dùng backend Agg để không mở cửa sổ
import matplotlib.pyplot as plt

# ====================================================
# 1. HELPER FUNCTIONS & VISUALIZATION
# ====================================================

def to_numpy(tensor):
    return tensor.cpu().detach().item()

def unnormalize(img_tensor):
    """
    Chuyển Tensor (ImageNet norm) về lại ảnh gốc RGB để vẽ
    """
    # [C, H, W] -> [H, W, C]
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Mean & Std chuẩn ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # De-normalize: x * std + mean
    img = img * std + mean
    
    # Clip về [0, 1] để hiển thị đúng màu
    return np.clip(img, 0, 1)

def visualize_prediction(img_tensor, mask_tensor, pred_tensor, save_path, iou_score, dice_score):
    """
    Vẽ và lưu ảnh so sánh: Gốc | Mask thật | Dự đoán chồng lớp
    Sử dụng matplotlib backend Agg để an toàn trên Linux.
    """
    try:
        orig_img = unnormalize(img_tensor)
        gt_mask = mask_tensor.squeeze().cpu().numpy()
        pred_mask = pred_tensor.squeeze().cpu().numpy()
        
        # Nếu pred_mask là logits (giá trị thực), chuyển về xác suất để vẽ đẹp hơn
        if pred_mask.max() > 1.5 or pred_mask.min() < -0.5:
            pred_mask = 1 / (1 + np.exp(-pred_mask)) # Sigmoid numpy

        # Tạo figure (khung vẽ)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        
        # 1. Ảnh gốc
        ax[0].imshow(orig_img)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        
        # 2. Ground Truth
        ax[1].imshow(gt_mask, cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[1].axis('off')
        
        # 3. Overlay Prediction
        ax[2].imshow(orig_img)
        # Mask thật: Màu xanh lá (Green), trong suốt
        ax[2].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap='Greens', alpha=0.3, vmin=0, vmax=1)
        
        # Dự đoán: Màu đỏ (Red), trong suốt, chỉ vẽ phần > 0.5
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        ax[2].imshow(np.ma.masked_where(pred_binary == 0, pred_binary), cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        
        ax[2].set_title(f"Pred Overlay\nIoU: {iou_score:.2f} | Dice: {dice_score:.2f}")
        ax[2].axis('off')
        
        # Lưu file và đóng ngay lập tức để giải phóng RAM
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    except Exception as e:
        print(f"[WARN] Error visualize: {e}")
    finally:
        # Bắt buộc đóng figure
        plt.close('all')

# ====================================================
# 2. METRICS (Dùng để đánh giá - Evaluation)
# ====================================================

def dice_coeff_hard(logits, target, threshold=0.5, epsilon=1e-6):
    """
    Dice Score 'cứng' (0 hoặc 1) dựa trên ngưỡng, dùng để báo cáo kết quả.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    # Tính trên toàn bộ batch (dimension [1, 2, 3] tương ứng C, H, W)
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice # Trả về vector [Batch_size]

def iou_core_hard(logits, target, threshold=0.5, epsilon=1e-6):
    """
    IoU Score 'cứng' để báo cáo kết quả.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou # Trả về vector [Batch_size]

# ====================================================
# 3. LOSS FUNCTIONS (Các Class Loss Clean & Optimized)
# ====================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Tính Dice Loss Soft (có thể đạo hàm được)
        probs = torch.sigmoid(logits)
        
        # Flatten [B, C, H, W] -> [B, -1]
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        """
        Khuyên dùng cho dữ liệu mất cân bằng (beta=0.7 để phạt nặng việc bỏ sót u)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        TP = (probs * targets).sum(dim=1)
        FP = ((1 - targets) * probs).sum(dim=1)
        FN = (targets * (1 - probs)).sum(dim=1)
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1.0 - tversky).mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma

    def forward(self, logits, targets):
        tversky = self.tversky(logits, targets)
        return torch.pow(tversky, self.gamma)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # BCE with Logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Tính pt
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal Term
        focal_term = (1 - pt).pow(self.gamma)
        
        # Alpha Term
        if self.alpha is not None:
            alpha_term = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_term * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss
            
        return loss.mean()

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.8, ce_ratio=0.5, focal_gamma=2.0):
        """
        Kết hợp: 50% Focal (Pixel) + 50% Dice (Cấu trúc)
        alpha=0.8: Ép model học lớp U nhiều hơn.
        """
        super().__init__()
        self.ce_ratio = ce_ratio
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        focal = self.focal(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_ratio * focal + (1 - self.ce_ratio) * dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
    
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice

# ====================================================
# 4. FACTORY FUNCTION (Hàm gọi Loss)
# ====================================================

def get_loss_function(loss_name):
    """
    Factory function để khởi tạo hàm loss dựa trên tên.
    """
    print(f"[INFO] Initializing Loss Function: {loss_name}")
    
    if loss_name == "Tversky_loss":
        return TverskyLoss(alpha=0.3, beta=0.7)
    
    elif loss_name == "FocalTversky_loss":
        return FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
    
    elif loss_name == "Combo_loss":
        # Alpha 0.8 để ưu tiên khối u (chiếm diện tích nhỏ)
        return ComboLoss(alpha=0.8, ce_ratio=0.5)
    
    elif loss_name == "Dice_loss":
        return DiceLoss()
        
    elif loss_name == "BCEDice_loss":
        return BCEDiceLoss()
    
    elif loss_name == "BCEw_loss":
        # Cảnh báo: Chỉ dùng khi bạn đã tính toán kỹ pos_weight
        # Ví dụ: 100 cho tỷ lệ u/bg = 1/100
        pos_weight = torch.tensor([100.0]).cuda() 
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    else:
        print(f"[WARN] Loss '{loss_name}' not found! Defaulting to Combo_loss.")
        return ComboLoss(alpha=0.8)
    
# ====================================================
# 5. SLIDING WINDOW INFERENCE UTILS (NEW)
# ====================================================

def adjust_gamma(image, gamma=1.0):
    """Hàm chỉnh Gamma nhanh bằng Lookup Table"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def predict_sliding_window(model, full_image_gray, device, window_size=512, stride=256):
    """
    Dự đoán trên ảnh gốc siêu lớn bằng cách trượt cửa sổ.
    Input: Ảnh xám gốc (H, W)
    Output: Mask dự đoán (H, W) (Giá trị 0-1)
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # 1. Chuẩn bị 3 kênh (Multi-View) ngay tại chỗ (On-the-fly)
    # Vì ảnh gốc rất nặng, ta không lưu sẵn 3 kênh ra đĩa mà tạo lúc chạy
    img_low = adjust_gamma(full_image_gray, gamma=0.5)
    img_high = adjust_gamma(full_image_gray, gamma=1.5)
    full_image_3c = np.stack([full_image_gray, img_low, img_high], axis=-1)
    
    h_orig, w_orig = full_image_3c.shape[:2]
    
    # 2. Padding để ảnh chia hết cho window_size
    pad_h = (window_size - h_orig % window_size) % window_size
    pad_w = (window_size - w_orig % window_size) % window_size
    
    image_padded = cv2.copyMakeBorder(full_image_3c, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_pad, w_pad = image_padded.shape[:2]
    
    # Map chứa tổng xác suất và map đếm số lần dự đoán
    prob_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    count_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    
    # Transform chuẩn hóa (giống lúc train)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    model.eval()
    
    # 3. Vòng lặp trượt
    # stride=256 nghĩa là overlap 50% -> Giúp biên dự đoán cực mượt
    with torch.no_grad():
        for y in range(0, h_pad - window_size + 1, stride):
            for x in range(0, w_pad - window_size + 1, stride):
                # Cắt patch
                patch = image_padded[y:y+window_size, x:x+window_size, :]
                
                # Preprocess
                tensor = transform(image=patch)['image'].unsqueeze(0).to(device)
                
                # Predict
                logits = model(tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                # Cộng dồn kết quả
                prob_map[y:y+window_size, x:x+window_size] += prob
                count_map[y:y+window_size, x:x+window_size] += 1.0

    # 4. Tính trung bình (Average Blending)
    final_prob = prob_map / np.maximum(count_map, 1.0)
    
    # Cắt bỏ phần padding thừa
    final_prob = final_prob[:h_orig, :w_orig]
    
    return final_prob