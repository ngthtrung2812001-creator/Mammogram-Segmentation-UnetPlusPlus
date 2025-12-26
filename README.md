# Mammogram Segmentation: Unet++ & EfficientNet-B4

Dự án phân đoạn khối u trên ảnh nhũ ảnh (Mammogram) sử dụng kiến trúc **Unet++** kết hợp với Encoder **EfficientNet-B4**. Hệ thống được tối ưu hóa cho dữ liệu y tế mất cân bằng nghiêm trọng (tỷ lệ u < 1%) bằng cách sử dụng các hàm loss chuyên dụng như **Tversky Loss** và **Focal Tversky Loss**.

## 📌 Tính năng nổi bật
* **Model mạnh mẽ:** Unet++ với backbone EfficientNet-B4 pre-trained trên ImageNet.
* **Loss Function chuyên dụng:** Tích hợp Tversky, Focal Tversky, Combo Loss để xử lý mất cân bằng dữ liệu.
* **Clean Architecture:** Code được tách biệt rõ ràng (Trainer, Dataset, Optimizer, Config).
* **Visualization:** Tự động vẽ biểu đồ Loss/Dice/IoU và xuất ảnh dự đoán trực quan sau khi test.
* **Hỗ trợ Linux Server:** Chạy tốt trên môi trường không màn hình (Headless) nhờ backend `Agg`.

## 🛠️ Cài đặt

1.  **Yêu cầu hệ thống:**
    * Python 3.8+
    * PyTorch (CUDA khuyến nghị)
    * Thư viện: `segmentation-models-pytorch`, `albumentations`, `pandas`, `matplotlib`, `opencv-python`.

2.  **Cài đặt thư viện:**
    ```bash
    pip install torch torchvision
    pip install segmentation-models-pytorch albumentations pandas matplotlib opencv-python imutils
    ```

## 📂 Cấu trúc Dữ liệu
Bạn cần sắp xếp dữ liệu theo cấu trúc sau để code tự động nhận diện:

```text
Dataset_Folder/
├── train/
│   ├── images/  (Chứa ảnh gốc .png/.jpg)
│   └── masks/   (Chứa ảnh mask tương ứng)
├── valid/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/