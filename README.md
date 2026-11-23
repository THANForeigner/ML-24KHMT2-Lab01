# Lab 01 — AI Impact on Students EDA

## Interaction Linear Regression — trạng thái hiện tại

Notebook: `notebooks/interaction_linear_regression.ipynb`; mã nguồn chính nằm trong `src/lab01_ml/features/engineering.py` và `src/lab01_ml/models/`. Các file chia sẵn trong `data/splits/` được kiểm tra với `data/assignments.csv` và `data/metadata.json`; không tự tạo hoặc ghi đè split. Metadata mô tả split hiện tại khác split EDA gốc, và xác nhận split chung của nhóm vẫn đang chờ.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/build_interaction_notebook.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/execute_interaction_notebook.py
```

Lệnh cuối chạy notebook trong kernel sạch bằng `nbclient`. Đầu ra và notebook đã chạy nằm trong `outputs/interaction/`. Khi chưa có xác nhận split chung, notebook chỉ thực hiện CV/validation, ghi rõ `final test: not executed` và không đọc target test. Chỉ sau khi nhóm thực sự xác nhận split và protocol, người chạy có thể đặt `$env:LAB_GROUP_SPLIT_APPROVED = 'yes'` trước lệnh thực thi để cho phép đánh giá test sau model lock; không dùng test chọn model. Với Kaggle/Colab, mount repo và dữ liệu chia sẵn, cài requirements rồi chạy hai script Python tương ứng từ thư mục gốc; không có GPU hay tải dataset mặc định.

Model joblib chỉ được tải từ tệp do chính nhóm tạo và tin cậy. Kết quả mô tả liên hệ trên dữ liệu synthetic, không chứng minh tác động nhân quả; thời điểm có sẵn của Anxiety/Dependency và các biến khác cần được xác minh.

Notebook chính: [notebooks/kaggle/01_eda.ipynb](notebooks/kaggle/01_eda.ipynb). Yêu cầu được triển khai theo [brief.md](brief.md).

Notebook tự đọc CSV thực tế trong **`archive/`**, không sửa file gốc, không giải nén hay tải dữ liệu khi chạy. Nếu có nhiều CSV phù hợp, đặt `DATA_PATH` ở cell cấu hình đầu tiên. Target là `Post_Semester_GPA`; `GPA_Change` chỉ dùng làm outcome phân tích.

## Chạy

Python 3.11+; môi trường đã kiểm thử được ghi trong `reports/run_metadata.json`.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/execute_notebook.py --verify-rerun
.venv\Scripts\python.exe scripts/render_eda_report.py
```

Lệnh `execute_notebook.py` dùng kernel riêng của `.venv`, chạy notebook hai lần trong kernel sạch, lưu outputs vào notebook chính và `reports/01_EDA.executed.ipynb`, kiểm tra artifacts/tính tái lập. Không cài kernelspec vào tài khoản người dùng. Lệnh `render_eda_report.py` dựng báo cáo HTML/Markdown từ artifacts vừa chạy; không fit model hoặc phân tích lại dữ liệu.

Cũng có thể mở notebook trong VS Code/Jupyter, chọn Python của `.venv`, rồi **Restart Kernel → Run All**. Cell cấu hình là nơi duy nhất thay đường dẫn, seed, ratios và save flags. Trên Kaggle/Colab: cài dependencies một lần ở môi trường nếu cần, đặt `DATA_PATH` tới CSV đã gắn; metadata trong notebook ghi rõ nguồn và các unknown, chạy EDA không cần kết nối mạng. Để dùng snapshot provenance bên ngoài project, gắn thêm `data/source_metadata.json` và đặt `SOURCE_METADATA_PATH`.

## Outputs

- `reports/EDA_REPORT.html`: báo cáo EDA đầy đủ 12 phần bằng tiếng Việt, nhúng biểu đồ để đọc offline, có nút In / Lưu PDF của trình duyệt.
- `reports/EDA_REPORT.md`: cùng nội dung dạng Markdown; hình/bảng liên kết tới artifacts.
- `reports/EDA_REPORT.metadata.json`: fingerprint của run và các artifacts dùng để dựng báo cáo.
- `reports/EDA_FINDINGS.md`: trả lời RQ01–RQ12, số liệu và liên kết evidence.
- `reports/DATA_DICTIONARY.md`: schema thực tế, miền/đơn vị/timing đã xác minh hoặc unknown.
- `reports/PREPROCESSING_PLAN.md`: quyết định từng feature và thiết kế M01–M05.
- `reports/tables/`, `reports/figures/`: thống kê CSV full precision và PNG 300 DPI.
- `reports/artifact_manifest.csv`: đủ T01–T48/F01–F57, status thật; aliases và skip reason.
- `reports/preprocessing_log.csv`: train median, reference category và constant drops.
- `reports/run_metadata.json`: fingerprint, config, versions, warnings, unresolved issues.
- `data/splits/`: row keys từ SHA-256 + original position; assignment không chứa target hay ID cá nhân.
- `reports/VERIFICATION.md` và `verification_results.json`: kết quả AC01–AC18 sau lệnh kiểm chứng.
- `reports/edge_case_verification.json`: kiểm tra thiếu cột, dữ liệu lỗi, constant target, VIF so với statsmodels và tính độc lập với target test. Có thể chạy lại bằng `.venv\Scripts\python.exe scripts/check_edge_cases.py`; fixtures tạm không được dùng làm bằng chứng EDA.

## Chính sách phân tích

Split mặc định 70/15/15, seed 42. Detailed EDA và diagnostics chỉ dùng **train**; model comparisons định trước dùng cùng **validation rows**. **Test chỉ counts/integrity**, giữ cho đánh giá cuối sau khi specification đóng băng. Student ID lặp hoặc trùng nội dung ngoài ID được group để tránh overlap; chỉ exact duplicates được xóa. Không impute target, không tự xóa outlier hợp lệ.

Metadata chính thức Kaggle xác nhận **synthetic**, publisher version 1, CC0-1.0; snapshot có bằng chứng trong `data/source_metadata.json`. Bản CSV cục bộ được định danh bằng SHA-256, chưa có publisher checksum xác nhận phiên bản byte. Timing của đa số biến chưa rõ nên E2/E3 và engineering comparison là **conditional / retrospective**, chưa phải deployment forecasts. Không kết luận AI gây tăng/giảm GPA.

Hai PDF được nhắc trong brief chưa có trong workspace. Chưa xác minh số thành viên và Ridge có được tính là một model bắt buộc theo rubric gốc hay không. Notebook ghi rõ các giới hạn này; không tạo dữ liệu hay thống kê thay thế.

## Rerun và thay cấu hình

Rerun cùng data/config ghi lại đúng artifacts do notebook sở hữu và giữ split ổn định. Nếu data/config khác, notebook báo lỗi và giữ outputs cũ. Chọn thư mục mới ở config hoặc qua biến môi trường:

```powershell
$env:LAB_OUTPUT_DIR = 'reports_alternative'
$env:LAB_SPLIT_DIR = 'data/splits_alternative'
$env:LAB_DATA_PATH = 'archive/ai_student_impact_dataset (1).csv'
.venv\Scripts\python.exe scripts/execute_notebook.py --verify-rerun
```

Không cần rebuild để sử dụng notebook. `scripts/build_notebook.py` là mã nguồn tạo bố cục/cells để bảo trì; chạy lại script này sẽ thay notebook bằng bản chưa execute, sau đó cần chạy lại lệnh kiểm chứng để lưu outputs mới.
