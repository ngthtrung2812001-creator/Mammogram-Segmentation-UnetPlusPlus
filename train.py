import argparse
import os
import torch
import numpy as np
import random
import segmentation_models_pytorch as smp

# Import các module vệ tinh (Đảm bảo bạn đã sửa các file này theo hướng dẫn trước)
from config import SEED, BASE_OUTPUT
from trainer import Trainer
from optimizer import get_optimizer # Hoặc tên hàm bạn đã sửa trong optimizer.py
from dataset import get_dataloaders
from result import export, export_evaluate
from utils import get_loss_function

def get_args():
    parser = argparse.ArgumentParser(description="Train, Pretrain hoặc Evaluate model AI")
    
    # --- THAM SỐ CƠ BẢN ---
    parser.add_argument("--mode", type=str, choices=["train", "pretrain", "evaluate"], required=True, help="Chế độ chạy")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn đến thư mục dataset")
    parser.add_argument("--epoch", type=int, default=50, help="Số epoch để train")
    
    # --- THAM SỐ MODEL & TRAINING ---
    parser.add_argument("--checkpoint", type=str, help="Đường dẫn file checkpoint (cho pretrain/eval)")
    parser.add_argument("--saveas", type=str, default="default_run", help="Tên thư mục lưu kết quả")
    parser.add_argument("--augment", action='store_true', help="Bật Augmentation (lật, xoay, nhiễu...)")
    
    # --- HYPERPARAMETERS ---
    parser.add_argument("--lr0", type=float, default=1e-4, help="Learning rate ban đầu")
    parser.add_argument("--batchsize", type=int, default=8, help="Kích thước Batch size")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay để chống overfitting")
    
    parser.add_argument("--img_size", type=int, nargs='+', default=[512, 512], help="Kích thước ảnh đầu vào [H, W]")
    parser.add_argument("--numclass", type=int, default=1, help="Số lớp output (Binary = 1)")
    
    # --- LỰA CHỌN LOSS & OPTIMIZER ---
    parser.add_argument("--loss", type=str, 
                        choices=["Tversky_loss", "FocalTversky_loss", "Combo_loss", "Dice_loss", "BCEw_loss", "BCEDice_loss"], 
                        default="Tversky_loss", 
                        help="Hàm loss sử dụng")
    
    parser.add_argument("--optimizer", type=str, choices=["Adam", "SGD", "AdamW"], default="AdamW", help="Optimizer sử dụng")
    
    args = parser.parse_args()
    
    # Kiểm tra logic bắt buộc
    if args.mode in ["pretrain", "evaluate"] and not args.checkpoint:
        parser.error(f"❌ Bạn phải cung cấp --checkpoint khi chạy chế độ '{args.mode}'")
        
    return args

def set_seed():
    """Thiết lập seed để tái lập kết quả"""
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):  
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING MODE: {args.mode.upper()}")
    print(f"📂 Dataset:  {args.data}")
    print(f"🧠 Model:    Unet++ (EfficientNet-B4)")
    print(f"📉 Loss:     {args.loss}")
    print(f"⚙️  Img Size: {args.img_size} | Batch: {args.batchsize} | LR: {args.lr0}")
    print(f"{'='*60}\n")

    set_seed()
    
    # ====================================================
    # 1. KHỞI TẠO MODEL (Unet++ & EfficientNet-B4)
    # ====================================================
    print(f"[INFO] Initializing Model Unet++ with EfficientNet-B4...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", # Đã đổi theo yêu cầu của bạn
        encoder_weights="imagenet",     
        in_channels=3,                  # Giả sử bạn copy kênh xám thành 3 kênh RGB
        classes=args.numclass,          # Thường là 1 cho Binary Segmentation
        decoder_attention_type="scse"   # Module attention giúp model tập trung vào vùng u
    )

    # ====================================================
    # 2. KHỞI TẠO OPTIMIZER
    # ====================================================
    # Lưu ý: Cần đảm bảo file optimizer.py có hàm nhận các tham số này
    optimizer = get_optimizer(
        model=model
        # Nếu hàm optimizer của bạn đã sửa để nhận lr/weight_decay thì bỏ comment dòng dưới:
         , lr=args.lr0, weight_decay=args.weight_decay, opt_name=args.optimizer
    ) 
    # Nếu chưa sửa optimizer.py, nó sẽ dùng mặc định trong config (không khuyến khích)

    # ====================================================
    # 3. KHỞI TẠO LOSS FUNCTION
    # ====================================================
    criterion = get_loss_function(args.loss)

    # ====================================================
    # 4. KHỞI TẠO TRAINER
    # ====================================================
    trainer = Trainer(
        model=model, 
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.epoch,
        patience=20 # Bạn có thể thêm tham số này vào args nếu muốn chỉnh
    )

    # ====================================================
    # 5. LOAD DỮ LIỆU (DATALOADERS)
    # ====================================================
    # Đảm bảo dataset.py/get_dataloaders đã sửa để nhận tham số
    trainLoader, validLoader, testLoader = get_dataloaders(
        data_dir=args.data,      
        batch_size=args.batchsize,
        img_size=args.img_size,  
        augment=args.augment
    )

    # ====================================================
    # 6. THỰC THI (TRAIN / PRETRAIN / EVALUATE)
    # ====================================================
    
    # --- CASE 1: TRAIN MỚI TỪ ĐẦU ---
    if args.mode == "train":
        print("[INFO] Start Training from scratch...")
        trainer.train(trainLoader, validLoader, resume_path=None)
        # Lưu kết quả vào thư mục định sẵn
        export(trainer, save_dir=args.saveas)

    # --- CASE 2: TRAIN TIẾP (PRETRAIN) ---
    elif args.mode == "pretrain":
        print(f"[INFO] Start Pre-training (Resume from {args.checkpoint})...")
        trainer.train(trainLoader, validLoader, resume_path=args.checkpoint)
        export(trainer, save_dir=args.saveas)

    # --- CASE 3: ĐÁNH GIÁ (EVALUATE) ---
    elif args.mode == "evaluate":
        print(f"[INFO] Start Evaluating...")
        
        # Tạo đường dẫn lưu ảnh visual
        visual_folder = os.path.join(BASE_OUTPUT, args.saveas, "prediction_images")
        
        # Chạy evaluate trên tập TEST (không phải valid)
        trainer.evaluate(
            test_loader=testLoader, 
            checkpoint_path=args.checkpoint,
            save_visuals=True,          
            output_dir=visual_folder    
        )
        
        # Xuất file CSV chi tiết
        export_evaluate(trainer, save_dir=args.saveas)

if __name__ == "__main__":
    args = get_args()
    main(args)