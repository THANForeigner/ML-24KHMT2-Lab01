# Chạy LAB / 01 bằng Docker Compose

## 1. Yêu cầu

- Docker Desktop, hoặc Docker Engine kèm Docker Compose v2.20 trở lên.
- Docker daemon đang chạy; có Internet trong lần build đầu.
- Chạy lệnh tại thư mục gốc dự án, nơi có `compose.yaml`.
- Có đầy đủ các file model trong `outputs/`, `data/metadata.json`, dữ liệu train và validation.

Máy chạy chỉ cần Docker; không cần cài Node.js, Python hay thư viện ML riêng. Container không train lại model.

## 2. Khởi động

```bash
docker compose up --build -d --wait
```

Lần đầu cần tải base image, cài dependencies và build React, nên có thể mất vài phút. Những lần sau Docker tái sử dụng build cache.

Mở:

- Tổng quan: <http://localhost:8080/#overview>
- Thử dự đoán: <http://localhost:8080/#prediction>
- So sánh: <http://localhost:8080/#comparison>

Giao diện và API dùng cùng một địa chỉ. Port `8080` trên máy được chuyển tới `8000` trong container. Bản chạy Python trực tiếp trên port `8000`, nếu có, không bị ảnh hưởng.

Kiểm tra trạng thái:

```bash
docker compose ps
docker compose logs --tail=100 app
curl http://localhost:8080/api/health
```

Health endpoint trả về `{"status":"ok"}`. Trạng thái `healthy` xác nhận API đã khởi động, không có nghĩa toàn bộ model đã được chạy thử.

## 3. Đổi port

Không cần tạo `.env` để chạy mặc định. File `.env.example` chỉ chứa cấu hình port mẫu; không ghi đè `.env` đang có.

Linux/macOS:

```bash
APP_PORT=8081 docker compose up -d --wait
```

PowerShell:

```powershell
$env:APP_PORT = "8081"
docker compose up -d --wait
```

Sau đó mở <http://localhost:8081>.

Hoặc thêm `APP_PORT=8081` vào file `.env` tại thư mục gốc. Compose chỉ dùng giá trị này để nội suy port; file `.env` không được mount hay chuyển nguyên nội dung vào container.

## 4. Dừng, khởi động lại và cập nhật

```bash
# Xem log trực tiếp; Ctrl+C chỉ thoát xem log
docker compose logs -f app

# Khởi động lại process, xóa cache model trong RAM
docker compose restart app

# Build lại sau khi sửa code, dependencies, dữ liệu hoặc model
docker compose up --build -d --wait

# Dừng và xóa container/network của ứng dụng này
docker compose down
```

`docker compose down` không xóa source hay artifact trên máy. Ứng dụng không cần database, volume dữ liệu hay dịch vụ Redis.

## 5. Kiểm thử đầy đủ qua HTTP

Nếu máy có Python, chạy script dùng thư viện chuẩn, không cần cài thêm gói:

```bash
python scripts/smoke_web.py
```

Nếu đổi port:

```bash
python scripts/smoke_web.py --url http://localhost:8081
```

Có thể dùng `python3` hoặc `.venv/bin/python` tùy môi trường. Script kiểm tra React build, 12 trường, sáu model thật, Simple M01–M05, đường nhạy cảm, chỉ số validation, scatter và việc không công khai file dữ liệu/`.env`.

## 6. Cấu trúc image

`Dockerfile` sử dụng hai stage:

1. **Frontend build:** Node 22, `npm ci` từ lockfile, build React.
2. **Runtime:** Python 3.13, dependencies khóa trong `requirements-docker.txt`, FastAPI phục vụ cả API lẫn React build.

Image runtime chỉ chứa source Python cần thiết, frontend đã build, model được chọn và dữ liệu train/validation có kiểm tra SHA-256. Không copy tập test, `.env`, Git history, notebook, báo cáo, `.venv` hoặc `node_modules`. `.dockerignore` dùng allowlist để các tệp không cần thiết không đi vào build context.

Container chạy với UID/GID `10001`, không phải root. Compose đặt filesystem chỉ đọc, cung cấp `/tmp` tạm trong RAM và bỏ Linux capabilities. Không bind-mount thư mục dự án.

Một worker Python và giới hạn luồng BLAS giúp tránh nhân bản cache model và tiêu thụ CPU không cần thiết. Chưa áp giới hạn RAM cứng; có thể xem mức dùng thực tế:

```bash
docker compose stats --no-stream
```

Không có bảo đảm image chạy vừa gói RAM 512 MB trước khi đo trên môi trường triển khai. File model vẫn có cảnh báo khác phiên bản scikit-learn; bộ smoke test kiểm tra dự đoán và metrics khớp với artifact hiện tại.

Khi chủ động nâng dependencies, cập nhật `docker/requirements.in`, tạo lại lock và chạy kiểm thử:

```bash
uv pip compile docker/requirements.in --python-version 3.13 --output-file requirements-docker.txt
docker compose up --build -d --wait
python scripts/smoke_web.py
```

## 7. Xử lý lỗi

| Lỗi | Cách xử lý |
| --- | --- |
| Không kết nối Docker daemon | Mở Docker Desktop hoặc khởi động Docker Engine |
| Port đã được dùng | Đổi `APP_PORT` theo mục 3 |
| `COPY` báo thiếu file | Kiểm tra artifact tương ứng đã có trong `outputs/`; không train tự động để thay thế |
| Container `unhealthy` | Xem `docker compose logs --tail=200 app` |
| Model lỗi hoặc điểm validation không khớp | Kiểm tra log, versions và đúng artifact; không ghi đè model gốc |
| Bị dừng do hết RAM | Xem `docker compose stats --no-stream` và tăng RAM Docker/máy chủ |

## 8. Đưa lên máy chủ

Compose mặc định chỉ mở port trên `127.0.0.1`, không công khai ứng dụng ra LAN/Internet. Để public, dùng reverse proxy có HTTPS, giới hạn request và chính sách truy cập phù hợp.

Render có thể build trực tiếp `Dockerfile`; không chạy file Compose này trên Render. Image hỗ trợ biến `PORT` do nền tảng cung cấp, bind `0.0.0.0` và có health endpoint `/api/health`. Việc đóng gói này chưa tự tạo dịch vụ hay phát sinh đăng ký trả phí.
