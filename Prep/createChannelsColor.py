import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

# --- CẤU HÌNH ---
INPUT_DIR = "/mnt/d/cbis_ddsm_512_lanczos/Patches/Train"  # Thư mục chứa ảnh gốc đã qua CLAHE q
OUT_LOW_DIR = os.path.join(INPUT_DIR, "gamma_low")   # Kênh 2: Làm sáng (Gamma < 1)
OUT_HIGH_DIR = os.path.join(INPUT_DIR, "gamma_high") # Kênh 3: Làm tối (Gamma > 1)

def create_dirs():
    os.makedirs(OUT_LOW_DIR, exist_ok=True)
    os.makedirs(OUT_HIGH_DIR, exist_ok=True)

def adjust_gamma(image, gamma=1.0):
    """
    Điều chỉnh Gamma cho ảnh.
    - gamma < 1: Ảnh sáng hơn (thấy chi tiết vùng tối).
    - gamma > 1: Ảnh tối hơn (nổi bật vùng sáng nhất).
    """
    invGamma = 1.0 / gamma
    # Tạo bảng lookup table (LUT) để map giá trị pixel nhanh
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_images():
    img_paths = glob.glob(os.path.join(INPUT_DIR, "images", "*.png"))
    create_dirs()
    
    print(f"🚀 Bắt đầu tạo kênh Đa Phơi Sáng (Multi-Exposure) cho {len(img_paths)} ảnh...")
    
    for img_path in tqdm(img_paths):
        filename = os.path.basename(img_path)
        
        # 1. Đọc ảnh gốc (đã CLAHE)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # --- KÊNH 2: GAMMA LOW (Sáng hơn) ---
        # Gamma 0.5 giúp làm nổi các rìa mờ của khối u
        img_low = adjust_gamma(img, gamma=0.5)
        
        # --- KÊNH 3: GAMMA HIGH (Tối hơn) ---
        # Gamma 1.5 giúp nhấn chìm nhiễu nền, chỉ giữ lại lõi trắng nhất của u
        img_high = adjust_gamma(img, gamma=1.5)
        
        # --- LƯU KẾT QUẢ ---
        cv2.imwrite(os.path.join(OUT_LOW_DIR, filename), img_low)
        cv2.imwrite(os.path.join(OUT_HIGH_DIR, filename), img_high)

    print(f"\n✅ HOÀN TẤT! Đã tạo xong bộ dữ liệu 3 kênh quang học.")
    print(f"- Channel 1: Ảnh gốc (CLAHE)")
    print(f"- Channel 2: Gamma Low (Làm sáng biên)")
    print(f"- Channel 3: Gamma High (Làm rõ lõi)")

if __name__ == "__main__":
    process_images()