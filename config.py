import torch
import os

# --- CHỈ GIỮ LẠI HẰNG SỐ CỐ ĐỊNH ---
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False

# Các giá trị mặc định (Fallback)
INIT_LR = 1e-4
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-4
INPUT_IMAGE_WIDTH = 512
INPUT_IMAGE_HEIGHT = 512
NUM_CLASSES = 1
NUM_EPOCHS = 50

# Optimizer params
BETA = (0.9, 0.999)
AMSGRAD = False