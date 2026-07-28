# Product Requirement Document (PRD): Data-Analyst Agent (DAA)

## 1. Tổng quan (Overview)
Data-Analyst Agent (DAA) là hệ thống phân tích và chuẩn hóa hiệu suất nội dung, giúp loại bỏ các yếu tố ngoại cảnh (nhiễu từ xu hướng) để tìm ra chất lượng cốt lõi của bài đăng. Hệ thống sử dụng kỹ thuật **Control Variates (Biến kiểm soát)** và cơ chế dự báo **Cold Start** để đánh giá công bằng cho cả nội dung cũ và mới.

## 2. Module Sinh dữ liệu mẫu (Mock Data Engine)
Hệ thống tự động tạo dữ liệu giả lập cho từng bài viết để phục vụ demo và kiểm thử luồng logic:
* **Numerical Mocking**: Sinh các chỉ số tương tác thô ($Reactions, Comments, Shares, Viewers\_75, Impressions$).

## 3. Workflow Xử lý dữ liệu Numerical (Trọng tâm)

Quy trình xác định hiệu suất thực chất của bài đăng thông qua 4 bước logic:

### Bước 1: Tính toán Hiệu suất thực tế ($Y$)
DAA tính toán tỷ lệ tương tác dựa trên trọng số hành động của người dùng:
* **Đối với Video**: $Y = \frac{1 \cdot Reactions + 3 \cdot Comments + 5 \cdot Shares + 2 \cdot Viewers\_75}{Impressions \cdot 11}$.
* **Đối với Bài viết**: $Y = \frac{1 \cdot Reactions + 3 \cdot Comments + 5 \cdot Shares}{Impressions \cdot 9}$.

### Bước 2: Xác định Điểm kỳ vọng ($X_i$) - Cơ chế Rẽ nhánh
Hệ thống thực hiện kiểm tra dữ liệu để xác định "điểm chấp" kỳ vọng cho bài đăng:

* **Trường hợp A: Database Lookup (Đã có lịch sử)**:
    * Nếu thể loại nội dung $i$ đã tồn tại trong Database, hệ thống tra cứu và lấy giá trị Trung bình tích lũy ($Cumulative\ Mean$) của thể loại đó làm $X_i$.
* **Trường hợp B: Cold Start Machine Learning (Thể loại mới)**:
    * Nếu thể loại $i$ chưa từng xuất hiện (ví dụ: Lần đầu đăng Reels), hệ thống kích hoạt luồng dự báo:
        1. **Chuẩn hóa**: Sử dụng `StandardScaler` để xử lý các đặc trưng ($Impressions, Engagement\_Rate$).
        2. **Dự báo**: Sử dụng mô hình `SGDRegressor` để ước lượng giá trị kỳ vọng.
* **Trường hợp C: Extreme Cold Start (Lịch sử rỗng hoặc có ít hơn 2 bài đăng)**:
    * Nếu tài khoản/kênh hoàn toàn mới, số lượng bài đăng $N < 2$, dữ liệu sẽ không đủ điều kiện thống kê để tính phương sai hoặc huấn luyện Machine Learning.
    * **Xử lý ngắt mạch (Fallback)**: Hệ thống bypass AI, lấy giá trị $Y$ làm baseline.

### Bước 3: Hiệu chỉnh Hiệu suất thực chất ($Y_{adj}$)
Sử dụng kỹ thuật **Control Variates** để triệt tiêu nhiễu xu hướng:
* **Công thức**: $Y_{adj} = Y - \theta \cdot (X_i - \mu_X)$.
* **Trong đó**:
    * $\theta$: Hệ số kiểm soát (tính bằng Hiệp phương sai / Phương sai lịch sử).
    * $\mu_X$: Trung bình tương tác của toàn bộ trang (Baseline toàn cục).

### Bước 4: Khoảng tin cậy và Ra quyết định
DAA tính toán Khoảng tin cậy ($CI$) tự động (Mặc định 95%) để đưa ra lệnh điều phối:
* **Phương sai tối ưu**: $Var(Y_{adj}) = Var(Y_{hist}) \times (1 - \rho^2)$.
* **Trong đó**: $\rho$ là hệ số tương quan Pearson giữa Lịch sử ($X_{hist}$) và Thực tế ($Y_{hist}$).
* **Logic rẽ nhánh**: 
    * Nếu Cận dưới $CI > \mu_X \rightarrow$ **Volume Up** (Tăng cường đầu tư).
    * Nếu Cận trên $CI < \mu_X \rightarrow$ **Volume Down** (Giảm bớt/Dừng).
    * Nếu $CI$ chứa $\mu_X \rightarrow$ **Inconclusive** (Chưa đủ cơ sở, cần theo dõi thêm).

## 4. Yêu cầu Công nghệ (Tech Stack)
* **Ngôn ngữ**: Python.
* **Xử lý số liệu**: Polars.
* **Machine Learning**: `scikit-learn` (`SGDRegressor`, `StandardScaler`).

## 5. Hướng dẫn sử dụng

Có thể cài đặt repository này như một Python Package và sử dụng bên trong các hệ thống lớn hơn.

### Cài đặt
Cài đặt trực tiếp qua thư mục local:
```bash
pip install -e .
```

### Cách gọi Agent trong code
```python
from data_analyst import DataAnalystAgent

# 1. Khởi tạo Agent
agent = DataAnalystAgent()

# 2. Chạy luồng phân tích (Hệ thống tự load mock_posts.csv)
result = agent.analyze()

# 3. Sử dụng Output trả về
print(result['decision'])            # Quyết định hành động (Volume Up/Down/Inconclusive)
print(result['y_adj'])               # Hiệu suất đã hiệu chỉnh
print(result['mu_x'])                # Baseline trung bình
print(result['confidence_interval']) # Khoảng tin cậy 95%
```
