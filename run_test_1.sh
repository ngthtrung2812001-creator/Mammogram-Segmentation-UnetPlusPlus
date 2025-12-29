# Lưu ý: Hãy thay đổi đường dẫn --data nếu thư mục chứa 7000 ảnh của bạn nằm ở chỗ khác.
python train.py --mode train \
    --data "/mnt/d/cbis_ddsm_512_lanczos" \
    --saveas "Run5_Offline_FullData" \
    --epoch 100 \
    --batchsize 8 \
    --img_size 512 512 \
    --lr0 5e-5 \
    --loss FocalTversky_loss \
    --optimizer AdamW \
    --weight_decay 1e-4

# ------------------------------------------------------------------------------   
# 3. ĐÁNH GIÁ MODEL (EVALUATE)
# Chạy trên tập Test để tính Dice/IoU và xuất ảnh dự đoán ra folder.
# Yêu cầu bắt buộc: Phải có tham số --checkpoint
# ------------------------------------------------------------------------------
python train.py --mode evaluate \
    --data "/mnt/d/cbis_ddsm_512_lanczos" \
    --checkpoint "output/Run5_Offline_FullData/best_dice_mass_model.pth" \
    --saveas "Eval_Run5_Offline_FullData" \
    --batchsize 16