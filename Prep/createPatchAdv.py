import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import random

# --- CẤU HÌNH GRAND MASTER ---
INPUT_IMG_DIR = "/mnt/d/DATAVALID/CBIS_ORG/PNG_mass_full/tinh_tinh/CBIS_DDSM/train/images"
INPUT_MASK_DIR = "/mnt/d/DATAVALID/CBIS_ORG/PNG_mass_full/tinh_tinh/CBIS_DDSM/train/masks"
OUTPUT_DIR = "/mnt/d/DATAVALID/CBIS_ORG/PNG_mass_full/tinh_tinh/Patches_Train"

PATCH_SIZE = 512
MIN_TUMOR_SIZE = 16    # Lọc nhiễu < 16px

# CẤU HÌNH OVERSAMPLING (Nhân bản dữ liệu)
NORMAL_OVERSAMPLE = 8  # U thường: Tạo 8 bản (1 chính + 7 lệch)
GIANT_OVERSAMPLE = 4   # U khổng lồ: Tạo 4 bản (dịch chuyển nhẹ)
NUM_NEGATIVES = 3      # Nền sạch: Tạo 3 bản

# Tạo thư mục
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

# --- CÁC HÀM BỔ TRỢ ---
def apply_clahe(image):
    """Tăng tương phản cục bộ"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def crop_with_padding(image, x, y, size):
    """Cắt ảnh an toàn, tự động thêm viền đen nếu vượt quá kích thước"""
    h, w = image.shape[:2]
    pad_l, pad_t, pad_r, pad_b = 0, 0, 0, 0
    
    if x < 0: pad_l, x = -x, 0
    if y < 0: pad_t, y = -y, 0
    if x + size > w: pad_r = x + size - w
    if y + size > h: pad_b = y + size - h
    
    crop = image[y:y+size, x:x+size]
    
    if pad_l or pad_r or pad_t or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    return crop

def find_body_mask(image):
    """Tìm vùng mô vú (loại bỏ nền đen và chữ nhiễu)"""
    _, thresh = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return thresh
    # Chỉ lấy đường bao lớn nhất (bầu ngực)
    max_cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [max_cnt], -1, (255), thickness=cv2.FILLED)
    return mask

# --- LOGIC XỬ LÝ CHÍNH ---

def process_giant_tumor(img, mask, x, y, w, h, base_name, idx):
    """Xử lý U > 512px: Zoom-out + Resize Lanczos + Giữ trọn vẹn"""
    max_dim = max(w, h)
    # Mở rộng 1.5 lần để bao trọn u và có chỗ dịch chuyển nhẹ
    crop_size = int(max_dim * 1.5)
    center_x, center_y = x + w//2, y + h//2
    
    # Chỉ dịch chuyển nhỏ (10%) để U luôn nằm trọn trong khung (KHÔNG CẮT CỤT)
    shift_limit = int(max_dim * 0.1)
    
    offsets = [(0,0)]
    for _ in range(GIANT_OVERSAMPLE - 1):
        offsets.append((random.randint(-shift_limit, shift_limit), 
                        random.randint(-shift_limit, shift_limit)))
    
    count = 0
    for i, (dx, dy) in enumerate(offsets):
        crop_x = center_x + dx - crop_size // 2
        crop_y = center_y + dy - crop_size // 2
        
        # Cắt vùng lớn
        img_large = crop_with_padding(img, crop_x, crop_y, crop_size)
        mask_large = crop_with_padding(mask, crop_x, crop_y, crop_size)
        
        # Resize chất lượng cao
        img_fin = cv2.resize(img_large, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LANCZOS4)
        mask_fin = cv2.resize(mask_large, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
        
        img_fin = apply_clahe(img_fin)
        
        cv2.imwrite(f"{OUTPUT_DIR}/images/{base_name}_T{idx}_GIANT_P{i}.png", img_fin)
        cv2.imwrite(f"{OUTPUT_DIR}/masks/{base_name}_T{idx}_GIANT_P{i}.png", mask_fin)
        count += 1
    return count

def process_normal_tumor(img, mask, x, y, w, h, base_name, idx):
    """Xử lý U <= 512px: Cắt Native + CHO PHÉP CẮT CỤT (Partial Cut)"""
    center_x, center_y = x + w//2, y + h//2
    
    # Dịch chuyển mạnh (50px) để mô phỏng cửa sổ trượt cắt vào u
    shift_limit = 75 
    
    offsets = [(0,0)]
    for _ in range(NORMAL_OVERSAMPLE - 1):
        offsets.append((random.randint(-shift_limit, shift_limit), 
                        random.randint(-shift_limit, shift_limit)))
        
    count = 0
    for i, (dx, dy) in enumerate(offsets):
        crop_x = center_x + dx - PATCH_SIZE // 2
        crop_y = center_y + dy - PATCH_SIZE // 2
        
        # Cắt trực tiếp (Nếu u lòi ra ngoài biên, hàm crop_with_padding sẽ tự xử lý)
        img_fin = crop_with_padding(img, crop_x, crop_y, PATCH_SIZE)
        mask_fin = crop_with_padding(mask, crop_x, crop_y, PATCH_SIZE)
        
        img_fin = apply_clahe(img_fin)
        
        cv2.imwrite(f"{OUTPUT_DIR}/images/{base_name}_T{idx}_NORM_P{i}.png", img_fin)
        cv2.imwrite(f"{OUTPUT_DIR}/masks/{base_name}_T{idx}_NORM_P{i}.png", mask_fin)
        count += 1
    return count

def main():
    img_paths = glob.glob(os.path.join(INPUT_IMG_DIR, "*.png")) # Hoặc *.jpg
    stats = {"giant": 0, "normal": 0, "neg": 0, "noise": 0}
    
    print(f"🚀 BẮT ĐẦU XỬ LÝ {len(img_paths)} ẢNH...")
    
    for img_path in tqdm(img_paths):
        filename = os.path.basename(img_path)
        base_name = filename[:-4]
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_path = os.path.join(INPUT_MASK_DIR, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros_like(img)
        
        if img is None: continue

        # 1. TÌM VÀ XỬ LÝ CÁC KHỐI U
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_tumors = [] # Lưu tọa độ u thật để tránh cắt trùng
        
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Lọc nhiễu
            if w < MIN_TUMOR_SIZE or h < MIN_TUMOR_SIZE:
                stats["noise"] += 1
                continue
            
            valid_tumors.append((x, y, w, h))
            
            # Phân loại và xử lý
            if w > PATCH_SIZE or h > PATCH_SIZE:
                stats["giant"] += process_giant_tumor(img, mask, x, y, w, h, base_name, idx)
            else:
                stats["normal"] += process_normal_tumor(img, mask, x, y, w, h, base_name, idx)

        # 2. XỬ LÝ NỀN (NEGATIVE)
        body_mask = find_body_mask(img)
        h_img, w_img = img.shape
        collected = 0
        attempts = 0
        
        while collected < NUM_NEGATIVES and attempts < 50:
            attempts += 1
            rx = random.randint(0, w_img - PATCH_SIZE)
            ry = random.randint(0, h_img - PATCH_SIZE)
            
            # Kiểm tra va chạm với u thật
            overlap = False
            for tx, ty, tw, th in valid_tumors:
                if (rx < tx+tw and rx+PATCH_SIZE > tx and ry < ty+th and ry+PATCH_SIZE > ty):
                    overlap = True; break
            if overlap: continue
            
            # Kiểm tra độ phủ thịt (>40%)
            if cv2.countNonZero(body_mask[ry:ry+PATCH_SIZE, rx:rx+PATCH_SIZE]) > (PATCH_SIZE**2 * 0.4):
                p_img = apply_clahe(img[ry:ry+PATCH_SIZE, rx:rx+PATCH_SIZE])
                p_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                cv2.imwrite(f"{OUTPUT_DIR}/images/{base_name}_NEG_{collected}.png", p_img)
                cv2.imwrite(f"{OUTPUT_DIR}/masks/{base_name}_NEG_{collected}.png", p_mask)
                collected += 1
                stats["neg"] += 1

    print(f"\n✅ HOÀN TẤT! Kết quả: {stats}")

if __name__ == "__main__":
    main()