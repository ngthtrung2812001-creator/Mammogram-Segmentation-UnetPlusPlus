# CHANGELOG - Mammogram Segmentation Project

Tài liệu này ghi lại tất cả những thay đổi quan trọng, sửa lỗi và cải tiến được thực hiện đối với source code gốc.

## [2025-12-27] - Refactoring & Optimization for Linux Server

### 🚨 Critical Fixes (Sửa lỗi nghiêm trọng)
* **Fix Circular Import:** Đã loại bỏ hoàn toàn lỗi "nhập vòng tròn" giữa `config.py` <-> `train.py` <-> `dataset.py`.
    * *Giải pháp:* Chuyển đổi kiến trúc từ phụ thuộc biến toàn cục (Global Config) sang Dependency Injection (Truyền tham số từ `main` xuống các hàm con).
* **Fix Linux Display Error:** Sửa lỗi crash khi sử dụng `matplotlib.pyplot` trên server Linux không có màn hình (Headless).
    * *Giải pháp:* Thêm backend `matplotlib.use('Agg')` vào đầu các file `utils.py` và `result.py`.
* **Fix Optimizer Logic:** Sửa lỗi `optimizer.py` không nhận tham số Learning Rate (`--lr0`) từ bàn phím mà luôn lấy giá trị mặc định.

### 🏗️ Architectural Changes (Thay đổi kiến trúc)
* **train.py (Main Controller):**
    * Đóng vai trò trung tâm điều phối.
    * Nhận toàn bộ Arguments (`--loss`, `--lr0`, `--data`,...) và phân phối xuống `trainer`, `dataset`, `optimizer`.
* **config.py:**
    * Loại bỏ logic xử lý `args`.
    * Chỉ còn giữ lại các hằng số tĩnh (`SEED`, `DEVICE`, `PIN_MEMORY`).
* **utils.py:**
    * Gộp chung các file metrics và loss rời rạc thành một module thống nhất.
    * Thêm `Factory Pattern` cho Loss Function (`get_loss_function`).

### 🚀 Model & Training Enhancements (Cải tiến mô hình)
* **Model Architecture:**
    * Nâng cấp từ `Unet` (EfficientNet-B3) lên **`UnetPlusPlus`** kết hợp Encoder **`EfficientNet-B4`** để tăng khả năng trích xuất đặc trưng.
* **Loss Functions:**
    * Tích hợp thêm **`TverskyLoss`** và **`FocalTverskyLoss`** chuyên trị dữ liệu mất cân bằng (tỷ lệ U < 1%).
    * Cập nhật `ComboLoss` với tham số `alpha=0.8` để ưu tiên học vùng khối u.
* **Validation Metrics:**
    * Thêm tính toán `Dice` và `IoU` tách biệt cho 2 trường hợp: Ảnh có bệnh (Mass) và Ảnh bình thường (Normal) để đánh giá trung thực hơn.

### 🛠️ Code Cleanup & Refactoring (Dọn dẹp code)
* **dataset.py:**
    * Hàm `get_dataloaders` giờ đây nhận trực tiếp `data_dir` và `img_size`.
    * Xóa bỏ các hardcoded paths cũ.
* **trainer.py:**
    * Loại bỏ `from config import *`.
    * Thêm cơ chế `try-except-finally` khi lưu ảnh visualize để đảm bảo đóng `plt.figure` và giải phóng RAM.
* **result.py:**
    * Sửa lỗi chính tả tên biến `csv_path_currrent` -> `csv_path`.
    * Thêm kiểm tra `os.path.exists` trước khi di chuyển file model để tránh lỗi crash khi file không tồn tại.

### 📉 Visualization
* Cập nhật hàm `visualize_prediction`:
    * Vẽ ảnh chồng lớp (Overlay) với độ trong suốt (Alpha blending) giúp dễ quan sát vị trí dự đoán so với nhãn gốc.
    * Tự động lưu ảnh ra đĩa thay vì cố gắng hiển thị (`plt.show()`) gây lỗi trên Server.

---
**Tác giả:** Gemini AI Assistant & User
**Môi trường khuyến nghị:** Linux Server (Ubuntu), Python 3.8+, PyTorch CUDA.