# Sleep Detection V3

Ứng dụng thử nghiệm nhận biết trạng thái tài xế từ ảnh và webcam bằng model
MobileNetV2 đã huấn luyện. Giao diện web sử dụng Gradio, hỗ trợ tiếng Việt và
chạy trực tiếp trên máy cá nhân.

Model trả về xác suất của ba lớp: **tỉnh táo** (`alert`), **buồn ngủ / mắt nhắm**
(`drowsy`) và **ngáp** (`yawning`). Repository bao gồm trọng số model, metadata,
ứng dụng web/desktop, notebook huấn luyện và các bài kiểm thử.

> Dự án phục vụ nghiên cứu và trình diễn. Dự đoán trên một khung hình không phải
> kết luận chắc chắn về tình trạng buồn ngủ và không thay thế thiết bị cảnh báo an toàn.

## Mục lục

- [Tính năng](#tính-năng)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu trúc repository](#cấu-trúc-repository)
- [Model và xử lý ảnh](#model-và-xử-lý-ảnh)
- [Notebook huấn luyện](#notebook-huấn-luyện)
- [Kiểm thử](#kiểm-thử)
- [Xử lý lỗi](#xử-lý-lỗi)
- [Giới hạn và quyền riêng tư](#giới-hạn-và-quyền-riêng-tư)
- [Đóng góp](#đóng-góp)
- [Giấy phép và nguồn tham khảo](#giấy-phép-và-nguồn-tham-khảo)

## Tính năng

- Tải ảnh và xem xác suất của cả ba lớp.
- Thử webcam với kết quả cập nhật khoảng mỗi 0,5 giây.
- Làm mượt tối đa năm dự đoán webcam gần nhất, riêng cho từng phiên sử dụng.
- Hiển thị **CHƯA ĐỦ CHẮC CHẮN** khi xác suất cao nhất dưới 55%.
- Hỗ trợ ứng dụng desktop bằng Tkinter và nhánh thử nghiệm crop mặt bằng YuNet.
- Kiểm tra checksum SHA-256, cú pháp source/notebook và dự đoán bằng model thật.

Không cần huấn luyện lại, tải dataset hoặc kết nối Google Drive để chạy ứng dụng.
Website chỉ lắng nghe tại `127.0.0.1` và không tạo liên kết chia sẻ công khai.

## Yêu cầu môi trường

- Windows 64-bit, **Python 3.13** và Python Launcher (`py`).
- Chrome hoặc Edge để sử dụng webcam; camera không bắt buộc nếu chỉ thử ảnh.
- Internet trong lần cài thư viện đầu tiên.

CPU đủ để chạy thử; không bắt buộc có GPU hoặc cài CUDA. Dependency chính
được ghim trong requirements: TensorFlow 2.21.0, OpenCV 5.0.0.93, NumPy 2.5.3
và Gradio 6.28.0. Môi trường đã được kiểm tra trên Windows/Python 3.13;
chưa xác nhận cài đặt mới trên macOS hoặc Linux.

## Cài đặt

### 1. Lấy source code

Clone repository hoặc chọn **Code → Download ZIP** trên GitHub rồi giải nén.
Mở PowerShell tại thư mục chứa `app.py` và `requirements.txt`.

Phải tải đầy đủ repository, bao gồm model và `class_names.json` trong
`versions/version_3/`. Không đổi cấu trúc thư mục và không sao chép môi trường
ảo từ máy khác.

### 2. Tạo môi trường và cài thư viện

Cài [Python 3.13 64-bit](https://www.python.org/downloads/windows/) kèm
Python Launcher, sau đó chạy:

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Các lệnh dùng trực tiếp Python trong môi trường ảo, không cần chạy `Activate.ps1`.

### 3. Kiểm tra và khởi động

```powershell
.\.venv\Scripts\python.exe .\tools\check_bundle.py
.\.venv\Scripts\python.exe .\app.py --check
.\.venv\Scripts\python.exe .\app.py
```

Truy cập [http://127.0.0.1:7860](http://127.0.0.1:7860).
Giữ cửa sổ PowerShell đang chạy; nhấn `Ctrl+C` để dừng server.
Những lần sau chỉ cần chạy lại lệnh `app.py`, không phải cài thư viện lại.

## Sử dụng

### Website

1. **Tải ảnh:** chọn ảnh trong mục **Tải ảnh**, sau đó bấm **Kiểm tra ảnh**.
2. **Webcam:** mở mục **Webcam**, cấp quyền camera và bấm nút ghi hình trong khung camera.
3. Đọc trạng thái và xác suất từng lớp. Trung bình năm dự đoán chỉ làm mượt
   hiển thị, không phải phép đánh giá buồn ngủ theo thời gian.

Các tùy chọn dòng lệnh:

```powershell
# Kiểm tra model và giao diện, không mở server
.\.venv\Scripts\python.exe .\app.py --check

# Đổi cổng nếu 7860 đang được sử dụng
.\.venv\Scripts\python.exe .\app.py --port 7861

# Không tự mở trình duyệt
.\.venv\Scripts\python.exe .\app.py --no-browser

# Xem các tham số hỗ trợ
.\.venv\Scripts\python.exe .\app.py --help
```

Với cổng 7861, truy cập `http://127.0.0.1:7861`. Địa chỉ localhost chỉ truy cập
ứng dụng trên chính máy đang chạy server, không phải link chia sẻ từ xa.

### Ứng dụng desktop

Thử ảnh hoặc camera bằng Tkinter với classifier toàn khung hình:

```powershell
.\.venv\Scripts\python.exe .\versions\version_3\desktop_test_app.py
```

Nhánh thử nghiệm phát hiện và crop khuôn mặt bằng YuNet:

```powershell
.\.venv\Scripts\python.exe .\versions\version_3\face_app.py
```

Crop mặt thay đổi vùng ảnh đầu vào nên có thể cho kết quả khác website.
Nhánh này không phải đường inference mặc định và chưa được chứng minh tốt hơn
classifier toàn khung hình.

## Cấu trúc repository

```text
sleep_detection/
├── app.py                          # Điểm khởi động website
├── requirements.txt                # Dependency web và desktop
├── final_training.ipynb            # Notebook Colab với output lịch sử
├── bundle_manifest.json            # SHA-256 của các file phân phối
├── apps/
│   └── version_3_web/
│       ├── app.py                  # Giao diện và inference Gradio
│       ├── requirements.txt
│       └── test_app.py             # Test model thật và callback
├── versions/
│   └── version_3/
│       ├── driver_drowsiness_cnn.keras
│       ├── class_names.json        # Metadata input và thứ tự lớp
│       ├── desktop_test_app.py
│       ├── face_app.py
│       ├── face_detection_yunet_2023mar.onnx
│       └── requirements-desktop.txt
├── tests/
│   └── test_bundle.py              # Test toàn vẹn và portability
├── tools/
│   └── check_bundle.py             # Kiểm tra checksum và cú pháp
├── licenses/
│   └── YuNet-LICENSE.txt
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── .gitignore
└── .gitattributes
```

Dataset, môi trường ảo và kết quả huấn luyện mới không được đưa vào repository.
Hai file trọng số đi kèm được cho phép track trong `.gitignore`.

## Model và xử lý ảnh

V3 dùng backbone MobileNetV2 với đầu ra phân loại ba lớp. Website và app desktop
mặc định xử lý **toàn ảnh**, không tự phát hiện/crop khuôn mặt.

1. Đưa ảnh về RGB, đổi kích thước thành `224×224`.
2. Giữ pixel trong khoảng `0–255` rồi đưa vào model.
3. Model tự chuẩn hóa qua lớp Rescaling đã tích hợp và trả ba xác suất.
4. Hiển thị lớp có xác suất cao nhất hoặc trạng thái chưa đủ chắc chắn.

Thứ tự output trong [class_names.json](versions/version_3/class_names.json):

- `0`: `alert` — tỉnh táo.
- `1`: `drowsy` — buồn ngủ / mắt nhắm.
- `2`: `yawning` — ngáp.

**Không chia ảnh thêm cho 255** vì sẽ làm sai hợp đồng đầu vào.
Giữ model và metadata đi cùng nhau khi thay đổi hoặc di chuyển file.

## Notebook huấn luyện

[final_training.ipynb](final_training.ipynb) là notebook Google Colab của lượt
huấn luyện trước, giữ nguyên code và output lịch sử. Notebook không bắt buộc
để chạy ứng dụng và không chạy bằng lệnh `python`.

Nguồn dữ liệu baseline:
[Open/Closed Eyes and Yawning Labelled](https://www.kaggle.com/datasets/aryansharma8911/open-closed-eyes-and-yawning-labelled).
Dataset không được phân phối trong repository.

Nếu muốn huấn luyện lại trên Colab:

1. Tạo bản sao notebook, chuẩn bị GPU và dung lượng Google Drive.
2. Kiểm tra đường dẫn lịch sử: workspace
   `/content/drive/MyDrive/programming/nckh`, dataset tại `driver_drowsiness/`
   và model đầu ra tại `Real-time-driver-drowsiness-detection/`.
3. Sửa mọi đường dẫn liên quan, kể cả cell dùng đường dẫn tuyệt đối, sang vùng
   huấn luyện riêng. Sao lưu model và checkpoint trước khi chạy.
4. Kiểm tra dữ liệu, mapping lớp và log từng bước. Cell chép dữ liệu có thao tác
   xóa cache `/content/cnn_dataset`; cell save có thể ghi đè artifact đầu ra.
5. Đánh giá và nạp lại model mới cùng metadata trước khi dùng trong ứng dụng.

Notebook tự cài dependency cho Colab; `requirements.txt` ở gốc chỉ phục vụ
inference/web, không phải lockfile của lượt training. Output lưu sẵn không
chứng minh một lượt chạy lại từ runtime sạch hay chất lượng trên camera mới.

## Kiểm thử

Chạy từ thư mục gốc sau khi cài dependency:

```powershell
.\.venv\Scripts\python.exe .\tools\check_bundle.py
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
.\.venv\Scripts\python.exe -m unittest apps.version_3_web.test_app -v
.\.venv\Scripts\python.exe .\app.py --check
```

Test bao gồm checksum/cú pháp, khởi động từ cwd khác, nạp YuNet, đầu ra model
thật, đối chiếu website/desktop, ảnh lỗi, ngưỡng tin cậy và lịch sử/reset theo phiên.
Hai bộ unittest hiện gồm **11 test** và đã đạt trên môi trường local.

Test tự động không thay thế thử webcam vật lý, cài sạch trên máy mới hoặc
đánh giá độ chính xác thực tế.

## Xử lý lỗi

- **Không nhận lệnh `py`:** kiểm tra Python 3.13 và Python Launcher đã được cài.
- **Thiếu model/metadata:** tải đầy đủ repo và giữ vị trí file trong `versions/version_3/`.
- **Cổng bận:** dùng `--port 7861` và mở đúng địa chỉ với cổng mới.
- **Webcam trống:** dùng Chrome/Edge, cấp quyền camera và đóng app khác đang giữ camera.
- **Lỗi DLL TensorFlow:** xem [hướng dẫn Windows của TensorFlow](https://www.tensorflow.org/install/pip#windows-native); không tải DLL từ nguồn không rõ.
- **Checksum không khớp:** kiểm tra file bị thiếu/thay đổi; không tạo lại manifest chỉ để bỏ qua lỗi.

## Giới hạn và quyền riêng tư

- V3 phân loại frame, không xác nhận sự kiện buồn ngủ kéo dài. Ngáp không đồng
  nghĩa buồn ngủ; mắt nhắm có thể chỉ là chớp mắt.
- Model không có lớp “không có mặt”; website vẫn có thể trả dự đoán cho ảnh không có mặt.
- Ánh sáng, nền, kính, góc mặt và camera khác nguồn training có thể làm dự đoán sai.
- Dataset baseline chưa chứng minh chia tách theo người/video; không suy ra khả
  năng tổng quát hóa chỉ từ kết quả trên split ảnh.
- Chỉ dùng webcam khi có sự đồng ý của người tham gia. App không có chức năng
  ghi video, nhưng frame được xử lý trên server local và có thể có cache tạm.

Không dùng prototype này như hệ thống bảo đảm an toàn khi lái xe hoặc công cụ chẩn đoán.

## Đóng góp

Giữ đúng hợp đồng input/output và chạy các lệnh kiểm thử khi sửa code.
Với thay đổi có chủ ý trên file phân phối, kiểm tra diff trước rồi cập nhật
manifest và chạy lại test:

```powershell
.\.venv\Scripts\python.exe .\tools\check_bundle.py --write-manifest
```

Không commit credential, môi trường ảo, dataset, ảnh webcam hoặc checkpoint
ngoài phạm vi thay đổi. Khi thay model, cập nhật metadata, tài liệu và test;
checksum không thay thế đánh giá chất lượng.

## Giấy phép và nguồn tham khảo

Thông báo MIT của source kế thừa được giữ trong [LICENSE](LICENSE).
YuNet có thông báo riêng tại [licenses/YuNet-LICENSE.txt](licenses/YuNet-LICENSE.txt).
Xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) để biết nguồn các thành phần.
Quyền phân phối dataset hoặc weights không tự suy ra từ giấy phép code.

- [Gradio — tài liệu chính thức](https://github.com/gradio-app/gradio/blob/main/README.md).
- [TensorFlow — hướng dẫn cài đặt](https://www.tensorflow.org/install/pip).
- [OpenCV Zoo — YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
