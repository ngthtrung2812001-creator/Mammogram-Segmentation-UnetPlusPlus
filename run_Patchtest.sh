#!/bin/bash

# ==============================================================================
# 🚀 GRAND MASTER PIPELINE RUNNER
# ==============================================================================

# 1. CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
# Thay đổi đường dẫn này trỏ đến thư mục "Final_Dataset_Patches" bạn đã tạo
DATA_PATH="/mnt/d/cbis_ddsm_512_lanczos/Patches"

# Tên thư mục sẽ lưu kết quả (trong folder output/z)
RUN_NAME="GrandMaster_EffB5_GammaV1"

# ==============================================================================
# PHẦN 1: HUẤN LUYỆN MỚI (TRAINING FROM SCRATCH)
# Chạy dòng này để bắt đầu train model.
# ==============================================================================

# echo "🔥 [START] Bắt đầu huấn luyện mô hình Grand Master..."

# python train.py \
#   --mode train \
#   --data "$DATA_PATH" \
#   --saveas "$RUN_NAME" \
#   --backbone "efficientnet-b5" \
#   --epoch 50 \
#   --batchsize 4 \
#   --lr0 1e-4 \
#   --loss FocalTversky_loss \
#   --optimizer AdamW \
#   --augment \
#   --img_size 512 512

# Giải thích tham số:
# --backbone "efficientnet-b5": Dùng mạng nơ-ron sâu và mạnh mẽ.
# --batchsize 4: An toàn cho GPU 8GB-12GB VRAM (Vì B5 rất nặng).
# --loss FocalTversky_loss: Tối ưu cho dữ liệu mất cân bằng (U nhỏ).
# --augment: Bật chế độ làm méo ảnh (Elastic Transform) để chống overfitting.

# ==============================================================================
# PHẦN 2: TIẾP TỤC HUẤN LUYỆN (RESUME / PRETRAIN)
# Dùng khi bị mất điện hoặc muốn train thêm epoch cho model cũ.
# (Bỏ comment dòng dưới để chạy)
# ==============================================================================

# echo "🔄 [RESUME] Tiếp tục huấn luyện từ checkpoint..."

# python train.py \
#   --mode pretrain \
#   --data "$DATA_PATH" \
#   --saveas "${RUN_NAME}_Resume" \
#   --checkpoint "output/$RUN_NAME/last_model.pth" \
#   --backbone "efficientnet-b5" \
#   --epoch 50 \
#   --batchsize 4 \
#   --lr0 1e-5 \
#   --loss FocalTversky_loss \
#   --augment

# ==============================================================================
# PHẦN 3: KIỂM NGHIỆM & ĐÁNH GIÁ (EVALUATE / TESTING)
# Chạy dòng này SAU KHI train xong để tính điểm Dice/IoU trên tập Test
# và xuất ảnh dự đoán ra để mắt thường kiểm tra.
# ==============================================================================

echo "📊 [EVAL] Đang đánh giá model tốt nhất trên tập Test..."

# python train.py \
#   --mode evaluate \
#   --data "$DATA_PATH" \
#   --saveas "$RUN_NAME" \
#   --checkpoint "output/$RUN_NAME/best_dice_mass_model.pth" \
#   --backbone "efficientnet-b5" \
#   --batchsize 8 \
#   --img_size 512 512

#Full image evaluation
python train.py \
  --mode evaluate \
  --data "$DATA_PATH" \
  --saveas "GrandMaster_FullEval" \
  --checkpoint "output/GrandMaster_EffB5_GammaV1/best_dice_mass_model.pth" \
  --backbone "efficientnet-b5" \
  --full_eval

# Lưu ý:
# --checkpoint: Trỏ vào file model có Dice Score cao nhất (best_dice_mass_model.pth)
# --batchsize 8: Lúc test không cần tính đạo hàm nên có thể tăng batchsize để chạy nhanh hơn.

echo "✅ [DONE] Hoàn tất quy trình!"