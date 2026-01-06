import argparse
import os
import torch
import numpy as np
import random
import segmentation_models_pytorch as smp

from config import SEED, BASE_OUTPUT
from trainer import Trainer
from optimizer import get_optimizer
# Import cả get_dataloaders VÀ FullImageDataset
from dataset import get_dataloaders, FullImageDataset
from result import export, export_evaluate
from utils import get_loss_function

def get_args():
    parser = argparse.ArgumentParser(description="Grand Master Training Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "pretrain", "evaluate"], default="train")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn folder Dataset")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--saveas", type=str, default="GrandMaster_Run")
    parser.add_argument("--augment", action='store_true', help="Bật Augmentation")
    
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, nargs='+', default=[512, 512])
    
    parser.add_argument("--backbone", type=str, default="efficientnet-b5")
    parser.add_argument("--loss", type=str, default="FocalTversky_loss")
    
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--full_eval", action='store_true', help="Bật chế độ đánh giá ảnh gốc")

    return parser.parse_args()

def set_seed():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

def main(args):  
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING MODE: {args.mode.upper()}")
    print(f"📂 Dataset:  {args.data}")
    print(f"🧠 Model:    Unet++ ({args.backbone})")
    print(f"📉 Loss:     {args.loss}")
    print(f"{'='*60}\n")

    set_seed()
    
    # 1. MODEL
    print(f"[INFO] Building Model...")
    model = smp.UnetPlusPlus(
        encoder_name=args.backbone,     
        encoder_weights="imagenet",     
        in_channels=3,
        classes=1,          
        decoder_attention_type="scse"
    )

    # 2. OPTIMIZER & LOSS
    optimizer = get_optimizer(model, lr=args.lr0, weight_decay=args.weight_decay, opt_name=args.optimizer)
    criterion = get_loss_function(args.loss)

    # 3. TRAINER
    trainer = Trainer(
        model=model, 
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.epoch,
        patience=15
    )

    # 4. EXECUTION
    if args.mode == "evaluate":
        print(f"[INFO] Start Evaluating...")
        
        if args.full_eval:
            print("[MODE] FULL IMAGE EVALUATION (SLIDING WINDOW)")
            # Load dataset ảnh gốc
            # Lưu ý: args.data ở đây phải trỏ vào folder test chứa subfolder images
            full_dataset = FullImageDataset(data_dir=args.data)
            
            output_dir = os.path.join(BASE_OUTPUT, args.saveas, "full_predictions")
            trainer.evaluate_full_images(
                test_dataset=full_dataset, 
                checkpoint_path=args.checkpoint,
                output_dir=output_dir
            )
        else:
            print("[MODE] PATCH-BASED EVALUATION")
            # Chỉ gọi get_dataloaders khi cần Patch
            _, _, testLoader = get_dataloaders(
                data_dir=args.data,      
                batch_size=args.batchsize,
                img_size=args.img_size,  
                augment=False
            )
            output_dir = os.path.join(BASE_OUTPUT, args.saveas, "patch_predictions")
            trainer.evaluate(testLoader, checkpoint_path=args.checkpoint, save_visuals=True, output_dir=output_dir)

    elif args.mode == "train":
        # Load data cho train
        trainLoader, validLoader, _ = get_dataloaders(args.data, args.batchsize, args.img_size, args.augment)
        if trainLoader is None: return # Dừng nếu không tìm thấy data
        
        print("[INFO] Start Training from scratch...")
        trainer.train(trainLoader, validLoader, resume_path=None)
        export(trainer, save_dir=args.saveas)

    elif args.mode == "pretrain":
        # Load data cho pretrain
        trainLoader, validLoader, _ = get_dataloaders(args.data, args.batchsize, args.img_size, args.augment)
        
        print(f"[INFO] Start Pre-training (Resume from {args.checkpoint})...")
        trainer.train(trainLoader, validLoader, resume_path=args.checkpoint)
        export(trainer, save_dir=args.saveas)

if __name__ == "__main__":
    args = get_args()
    main(args)