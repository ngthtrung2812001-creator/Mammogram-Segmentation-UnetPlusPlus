#!/bin/bash

# ==============================================================================
# 🚀 KỊCH BẢN KIỂM NGHIỆM TRÊN ẢNH GỐC (FULL SLIDING WINDOW)
# ==============================================================================

# 1. ĐƯỜNG DẪN DỮ LIỆU (ĐÃ SỬA)
# Code sẽ tự động tìm folder 'images' bên trong đường dẫn này.
# Dựa trên ảnh chụp màn hình folder Patches của bạn, đường dẫn đúng phải có thêm /test
DATA_PATH1="/mnt/d/cbis_ddsm_512_lanczos/Patches/test"

# 2. TÊN MODEL VÀ KẾT QUẢ
RUN_NAME="GrandMaster_EffB5_BCEDice"

# Đường dẫn đến file model (Kiểm tra kỹ lại xem file này có tồn tại không)
CHECKPOINT="output/GrandMaster_EffB5_BCEDice/best_dice_mass_model.pth"

#!/bin/bash

# --- CẤU HÌNH TRAIN LẠI TỪ ĐẦU ---
# Đổi tên saveas để không ghi đè model cũ (để so sánh nếu cần

# Trỏ vào folder chứa Patches (Train/Val/Test)
DATA_PATH="/mnt/d/cbis_ddsm_512_lanczos/Patches" 

echo "🔥 [START] Retraining with BCEDice Loss..."

python train.py \
  --mode train \
  --data "$DATA_PATH" \
  --saveas "$RUN_NAME" \
  --epoch 50 \
  --batchsize 8 \
  --img_size 512 512 \
  --lr0 1e-4 \
  --loss BCEDice_loss \
  --optimizer AdamW \
  --backbone "tu-resnest50d" \
  --augment

echo "✅ [DONE] Training started. Output: output/$RUN_NAME"
# ==============================================================================
# 3. LỆNH THỰC THI
# ==============================================================================

echo "🔥 [START] Đang quét Sliding Window trên tập ảnh gốc..."
echo "📂 Dữ liệu đầu vào: $DATA_PATH1"
echo "   (Code sẽ tìm ảnh tại: $DATA_PATH1/images/*.png)"
echo "🧠 Model checkpoint: $CHECKPOINT"

python train.py \
  --mode evaluate \
  --data "$DATA_PATH1" \
  --saveas "$RUN_NAME" \
  --checkpoint "$CHECKPOINT" \
  --backbone "tu-resnest50d" \
  --full_eval \
  --batchsize 1

# LƯU Ý:
# Nếu vẫn báo 0 ảnh, hãy kiểm tra xem ảnh của bạn là đuôi .png hay .jpg
# và sửa trong file dataset.py dòng glob(...) tương ứng.

echo "✅ [DONE] Kiểm tra kết quả tại: output/$RUN_NAME/full_predictions"