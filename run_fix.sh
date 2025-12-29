# ==============================================================================
# HƯỚNG DẪN CHẠY LỆNH (COMMAND LINE CHEATSHEET)
# Lưu ý: Thay thế đường dẫn "--data" bằng đường dẫn thực tế trên máy bạn.
# ==============================================================================
DATA_PATH="/mnt/d/cbis_ddsm_512_lanczos"
# ------------------------------------------------------------------------------
# 1. HUẤN LUYỆN MỚI (TRAIN FROM SCRATCH)
# ------------------------------------------------------------------------------

# > Cấu hình cơ bản (Mặc định dùng Tversky Loss, AdamW)
#python train.py --mode train --data "$DATA_PATH" --saveas "Run1_Basic" --epoch 50 --batchsize 8 --img_size 512 512

# > Cấu hình nâng cao (Bật Augmentation, giảm LR, dùng Focal Tversky cho data khó)
#ython train.py --mode train --data "$DATA_PATH" --saveas "Run2_Advanced_3" --epoch 100 --batchsize 8 --lr0 1e-4 --augment --loss FocalTversky_loss --optimizer AdamW

# > Thử nghiệm với Combo Loss (Kết hợp Dice + Focal)
python train.py --mode train --data "$DATA_PATH" --saveas "Run3_Combo" --epoch 100 --batchsize 8 --img_size 512 512 --lr0 1e-4 --loss Combo_loss --optimizer AdamW


# ------------------------------------------------------------------------------
# 2. HUẤN LUYỆN TIẾP (PRETRAIN / RESUME)
# Dùng khi bị ngắt điện, crash hoặc muốn train thêm epoch từ model cũ.
# Yêu cầu bắt buộc: Phải có tham số --checkpoint
# ------------------------------------------------------------------------------

# > Tiếp tục train từ model Run1 (Ví dụ train thêm 50 epoch nữa)
#python train.py --mode pretrain --data "D:/ISIC_dataset/format_dataset" --checkpoint "output/Run1_Basic/last_model.pth" --saveas "Run1_Resume" --epoch 50 --lr0 1e-5


# ------------------------------------------------------------------------------
# 3. ĐÁNH GIÁ MODEL (EVALUATE)
# Chạy trên tập Test để tính Dice/IoU và xuất ảnh dự đoán ra folder.
# Yêu cầu bắt buộc: Phải có tham số --checkpoint
# ------------------------------------------------------------------------------

# > Đánh giá model tốt nhất của Run2
python train.py --mode evaluate --data "$DATA_PATH" --checkpoint "output/Run3_Combo/best_dice_mass_model.pth" --saveas "Eval_Run3_Combo" --batchsize 16

# ------------------------------------------------------------------------------
# GIẢI THÍCH THAM SỐ QUAN TRỌNG:
# --mode:       Chế độ chạy (train / pretrain / evaluate).
# --data:       Đường dẫn thư mục dataset (chứa subfolders train/valid/test).
# --saveas:     Tên thư mục sẽ tạo trong folder 'output' để lưu kết quả.
# --loss:       Tên hàm loss (Tversky_loss, FocalTversky_loss, Combo_loss, Dice_loss).
# --checkpoint: Đường dẫn file .pth (Bắt buộc nếu mode là pretrain hoặc evaluate).
# --augment:    Cờ bật tăng cường dữ liệu (Xoay, lật, chỉnh sáng...). Chỉ cần gõ --augment là bật.
# --img_size:   Kích thước ảnh đầu vào [Cao Rộng]. Ví dụ: 512 512.
# ------------------------------------------------------------------------------