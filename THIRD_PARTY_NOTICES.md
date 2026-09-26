# Nguồn và thông báo giấy phép

- `LICENSE` giữ nguyên thông báo MIT có trong V3 gốc (Yagagiri Manideep).
  Không xóa attribution của source kế thừa.
- `face_detection_yunet_2023mar.onnx`: bản YuNet có sẵn trong workspace V3,
  sao chép nguyên bytes. Thông báo MIT upstream được giữ tại `licenses/YuNet-LICENSE.txt`.
  Nguồn: [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  và [giấy phép upstream](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE).
  Manifest ghi hash bản local; chưa đối chiếu bytes với trọng số upstream.
- TensorFlow, OpenCV, NumPy và Gradio được cài từ pip; không chép package/vendor vào repo.
  Mỗi dependency giữ giấy phép riêng.
- Model V3 và notebook dùng dataset Kaggle
  `aryansharma8911/open-closed-eyes-and-yawning-labelled`; không phân phối ảnh dataset.
  Bộ này không xác nhận quyền tái phân phối dữ liệu/weights phát sinh từ dataset.
  Trước khi public hoặc dùng thương mại, chủ dự án cần kiểm tra điều khoản nguồn
  dữ liệu, pretrained weights và quyền của các thành phần mình đưa lên.

Chỉ sao chép source/model không tạo ra giấy phép mới cho dữ liệu hoặc thay thế
việc xem xét quyền của chủ dự án.
