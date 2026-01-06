import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from imutils import paths
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob # Cần thêm thư viện glob

from config import SEED, PIN_MEMORY

# ====================================================
# 1. CLASS DATASET CHO PATCHES (TRAINING)
# ====================================================
class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]

        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Đọc mask
        mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE) 

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            
            # Threshold lại cho chắc chắn
            mask = (mask > 127).float()
            mask = mask.unsqueeze(0) 

        return image, mask, imagePath

# ====================================================
# 2. CLASS DATASET CHO ẢNH GỐC (FULL EVALUATION) -> ĐÃ THÊM MỚI
# ====================================================
class FullImageDataset(Dataset):
    def __init__(self, data_dir):
        # Tìm ảnh trong subfolder 'images' của đường dẫn data_dir
        self.image_paths = sorted(glob(os.path.join(data_dir, "images", "*.png")))
        self.mask_paths = sorted(glob(os.path.join(data_dir, "masks", "*.png")))
        
        # Nếu không có mask thì tạo list None
        if len(self.mask_paths) == 0:
            self.mask_paths = [None] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Đọc ảnh gốc (1 kênh xám) - KHÔNG RESIZE
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        mask = None
        # Kiểm tra an toàn nếu có mask
        if idx < len(self.mask_paths) and self.mask_paths[idx] is not None:
            if os.path.exists(self.mask_paths[idx]):
                m = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
                mask = (m > 127).astype(np.float32)
        
        return image, mask, img_path

# ====================================================
# 3. HÀM HỖ TRỢ REPRODUCIBILITY
# ====================================================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ====================================================
# 4. HÀM TẠO DATALOADERS
# ====================================================
def get_dataloaders(data_dir, batch_size, img_size, augment=False):
    height, width = img_size[0], img_size[1]
    
    # --- A. ĐỊNH NGHĨA TRANSFORMS ---
    base_transform = [
        A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    if augment:
        print(f"[INFO] Using HEAVY Data Augmentation for Training")
        train_ops = [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.GaussianBlur(blur_limit=3, p=1),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        train_transform = A.Compose(train_ops)
    else:
        train_transform = A.Compose(base_transform)

    valid_transform = A.Compose(base_transform)

    # --- B. TẠO ĐƯỜNG DẪN DỮ LIỆU ---
    # data_dir ở đây là folder GỐC (chứa train/valid/test)
    train_img_dir = os.path.join(data_dir, "train", "images")
    train_msk_dir = os.path.join(data_dir, "train", "masks")
    valid_img_dir = os.path.join(data_dir, "val", "images")
    valid_msk_dir = os.path.join(data_dir, "val", "masks")
    test_img_dir = os.path.join(data_dir, "test", "images")
    test_msk_dir = os.path.join(data_dir, "test", "masks")

    # Kiểm tra folder train có tồn tại không để tránh lỗi
    if not os.path.exists(train_img_dir):
        # Nếu không có folder train, có thể người dùng đang chỉ muốn test
        # Trả về None cho train/valid loader để code không crash ngay lập tức
        print(f"⚠️ CẢNH BÁO: Không tìm thấy folder train tại {train_img_dir}")
        return None, None, None

    trainImagesPaths = sorted(list(paths.list_images(train_img_dir)))
    trainMasksPaths  = sorted(list(paths.list_images(train_msk_dir)))
    validImagesPaths = sorted(list(paths.list_images(valid_img_dir)))
    validMasksPaths  = sorted(list(paths.list_images(valid_msk_dir)))
    testImagesPaths  = sorted(list(paths.list_images(test_img_dir)))
    testMasksPaths   = sorted(list(paths.list_images(test_msk_dir)))
    
    # --- C. KHỞI TẠO DATASET ---
    trainDS = SegmentationDataset(trainImagesPaths, trainMasksPaths, transforms=train_transform)
    validDS = SegmentationDataset(validImagesPaths, validMasksPaths, transforms=valid_transform)
    testDS  = SegmentationDataset(testImagesPaths, testMasksPaths, transforms=valid_transform)
    
    print(f"[INFO] Found {len(trainDS)} training images")
    print(f"[INFO] Found {len(validDS)} validation images")
    print(f"[INFO] Found {len(testDS)} test images")

    # --- D. KHỞI TẠO DATALOADER ---
    g = torch.Generator()
    g.manual_seed(SEED)

    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=batch_size, pin_memory=PIN_MEMORY, num_workers=4, worker_init_fn=seed_worker, generator=g)
    validLoader = DataLoader(validDS, shuffle=False, batch_size=batch_size, pin_memory=PIN_MEMORY, num_workers=4, worker_init_fn=seed_worker, generator=g)
    testLoader = DataLoader(testDS, shuffle=False, batch_size=batch_size, pin_memory=PIN_MEMORY, num_workers=4, worker_init_fn=seed_worker, generator=g)
    
    return trainLoader, validLoader, testLoader