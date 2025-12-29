python train.py --mode train \
    --data "/mnt/d/cbis_ddsm_512_lanczos" \
    --saveas "Run4_AntiOverfit" \
    --epoch 100 \
    --batchsize 8 \
    --img_size 512 512 \
    --lr0 5e-5 \
    --weight_decay 1e-2 \
    --augment \
    --loss FocalTversky_loss \
    --optimizer AdamW

# ------------------------------------------------------------------------------   
# 3. ĐÁNH GIÁ MODEL (EVALUATE)
# Chạy trên tập Test để tính Dice/IoU và xuất ảnh dự đoán ra folder.
# Yêu cầu bắt buộc: Phải có tham số --checkpoint
# ------------------------------------------------------------------------------             
python train.py --mode evaluate --data "/mnt/d/cbis_ddsm_512_lanczos" --checkpoint "output/Run4_AntiOverfit/best_dice_mass_model.pth" --saveas "Eval_Run4_AntiOverfit" --batchsize 16