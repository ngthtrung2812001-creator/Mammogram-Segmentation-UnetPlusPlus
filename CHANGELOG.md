# CHANGELOG - Project Mammogram Segmentation (Grand Master Edition)

## 📦 Version 3.0.0 (Grand Master) - Current Stable
**Ngày phát hành:** 2024-05-20
**Mục tiêu:** Tối ưu hóa toàn diện cho bài toán phân đoạn khối u vú (Mass Segmentation) trên dữ liệu DDSM nhiễu.

### 🚀 Tính năng mới (New Features)
* **Multi-View Input (3 Channels):**
    * Thay thế input ảnh xám đơn thuần bằng chồng ảnh 3 lớp (Stacking):
        1.  **Red Channel:** Ảnh gốc xử lý CLAHE (Contrast Limited Adaptive Histogram Equalization).
        2.  **Green Channel:** Gamma Low (γ=0.5) - Làm sáng vùng tối để lộ diện chân rết/tua gai (Spiculations).
        3.  **Blue Channel:** Gamma High (γ=1.5) - Làm tối nền để nổi bật lõi khối u đậm đặc.
* **Dynamic Patch Generation Strategy:**
    * **U Thường (≤512px):** Cắt ngẫu nhiên có độ lệch (Random Shift) để mô phỏng cửa sổ trượt.
    * **U Khổng lồ (>512px):** Chiến thuật "Zoom-out" (1.5x context) + Resize Lanczos4 để giữ trọn vẹn hình thái học.
* **Model Architecture Upgrade:**
    * Nâng cấp Backbone từ `EfficientNet-B4` lên **`EfficientNet-B5`** (Pre-trained ImageNet).
    * Sử dụng **U-Net++ (Nested U-Net)** với Attention Decoder (`scse`).
* **Advanced Augmentation (Online):**
    * Tích hợp `Albumentations` với **Elastic Transform** & **Grid Distortion** để mô phỏng tính chất đàn hồi của mô mềm.

### 🛠️ Sửa lỗi & Cải tiến (Bug Fixes & Improvements)
* **FIXED:** Loại bỏ hoàn toàn phương pháp SAM (Segment Anything Model) do hiện tượng "Over-smoothing" (mất gai) và "Hallucination" (bắt nhầm nhiễu CLAHE).
* **FIXED:** Loại bỏ phương pháp Canny/Sobel Edge do quá nhạy với nhiễu hạt của ảnh X-quang.
* **IMPROVED:** Sử dụng **Focal Tversky Loss** để giải quyết triệt để vấn đề mất cân bằng dữ liệu (Class Imbalance).

---

## 📦 Version 2.0.0 (Experimental) - Deprecated
**Trạng thái:** Đã hủy bỏ (Failed experiments)
* Thử nghiệm tích hợp SAM (`vit_h`) để tạo Mask gợi ý. -> **Thất bại** (Mask bị vo tròn, mất chi tiết gai).
* Thử nghiệm kênh cạnh (Edge Channels) dùng Sobel. -> **Thất bại** (Nhiễu quá nhiều do CLAHE).

## 📦 Version 1.0.0 (Legacy)
**Trạng thái:** Dự án gốc
* Input: Ảnh xám 1 kênh (Grayscale).
* Model: U-Net cơ bản hoặc U-Net++ (Backbone nhỏ).
* Loss: Dice Loss cơ bản.
* Nhược điểm: Hay bị dương tính giả (False Positive) ở vùng mô đặc và bỏ sót các khối u lớn.