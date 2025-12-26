import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from imutils import paths
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Chỉ import hằng số cấu hình hệ thống, không import logic
from config import SEED, PIN_MEMORY

# ====================================================
# 1. CLASS DATASET
# ====================================================
class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # 1. Lấy đường dẫn
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]

        # 2. Đọc ảnh và mask
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Đọc mask ở chế độ Grayscale
        mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE) 

        # 3. Áp dụng Albumentations (Augmentation)
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            
            # 4. Xử lý Mask sau khi augment
            # Mask cần chuyển về 0.0 và 1.0 (float32)
            # Resize của Albumentations đôi khi trả về giá trị nội suy, nên cần threshold lại cho chắc chắn
            mask = (mask > 127).float()
            
            # Thêm chiều kênh (Channel) cho mask: [H, W] -> [1, H, W]
            mask = mask.unsqueeze(0) 

        return image, mask, imagePath

# ====================================================
# 2. HÀM HỖ TRỢ REPRODUCIBILITY
# ====================================================
def seed_worker(worker_id):
    """Đảm bảo tính ngẫu nhiên giống nhau mỗi lần chạy trên nhiều workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ====================================================
# 3. HÀM TẠO DATALOADERS (TRUNG TÂM)
# ====================================================
def get_dataloaders(data_dir, batch_size, img_size, augment=False):
    """
    Hàm tạo Dataloader nhận tham số trực tiếp, không phụ thuộc Config cứng.
    """
    # img_size là list [H, W]
    height, width = img_size[0], img_size[1]
    
    # --- A. ĐỊNH NGHĨA TRANSFORMS ---
    # Transform cơ bản (luôn dùng cho Valid/Test)
    base_transform = [
        A.Resize(
            height=height, 
            width=width, 
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    if augment:
        print("[INFO] Using Data Augmentation for Training")
        # Augmentation chỉ áp dụng cho Train
        train_ops = [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=0, border_mode=cv2.BORDER_REFLECT_101, p=0.2),
            # Normalize và ToTensor luôn ở cuối
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        train_transform = A.Compose(train_ops)
    else:
        print("[INFO] No Augmentation")
        train_transform = A.Compose(base_transform)

    valid_transform = A.Compose(base_transform)

    # --- B. TẠO ĐƯỜNG DẪN DỮ LIỆU ---
    # Giả sử cấu trúc thư mục chuẩn: data_dir/train/images, data_dir/train/masks,...
    train_img_dir = os.path.join(data_dir, "train", "images")
    train_msk_dir = os.path.join(data_dir, "train", "masks")
    
    valid_img_dir = os.path.join(data_dir, "valid", "images")
    valid_msk_dir = os.path.join(data_dir, "valid", "masks")
    
    test_img_dir = os.path.join(data_dir, "test", "images")
    test_msk_dir = os.path.join(data_dir, "test", "masks")

    # Lấy danh sách file (đã sort để đảm bảo khớp cặp ảnh-mask)
    trainImagesPaths = sorted(list(paths.list_images(train_img_dir)))
    trainMasksPaths  = sorted(list(paths.list_images(train_msk_dir)))

    validImagesPaths = sorted(list(paths.list_images(valid_img_dir)))
    validMasksPaths  = sorted(list(paths.list_images(valid_msk_dir)))

    testImagesPaths  = sorted(list(paths.list_images(test_img_dir)))
    testMasksPaths   = sorted(list(paths.list_images(test_msk_dir)))

    # Kiểm tra sơ bộ
    if len(trainImagesPaths) == 0:
        print(f"⚠️ CẢNH BÁO: Không tìm thấy ảnh train nào tại {train_img_dir}")
    
    # --- C. KHỞI TẠO DATASET ---
    trainDS = SegmentationDataset(trainImagesPaths, trainMasksPaths, transforms=train_transform)
    validDS = SegmentationDataset(validImagesPaths, validMasksPaths, transforms=valid_transform)
    testDS  = SegmentationDataset(testImagesPaths, testMasksPaths, transforms=valid_transform)
    
    print(f"[INFO] Found {len(trainDS)} training images")
    print(f"[INFO] Found {len(validDS)} validation images")
    print(f"[INFO] Found {len(testDS)} test images")

    # --- D. KHỞI TẠO DATALOADER ---
    # Generator để kiểm soát tính ngẫu nhiên của shuffle
    g = torch.Generator()
    g.manual_seed(SEED)

    trainLoader = DataLoader(
        trainDS, 
        shuffle=True,
        batch_size=batch_size, 
        pin_memory=PIN_MEMORY,
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )
    
    validLoader = DataLoader(
        validDS, 
        shuffle=False,
        batch_size=batch_size, 
        pin_memory=PIN_MEMORY,
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )
    
    testLoader = DataLoader(
        testDS, 
        shuffle=False,
        batch_size=batch_size, 
        pin_memory=PIN_MEMORY,
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )
    
    return trainLoader, validLoader, testLoader