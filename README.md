# Product Requirement Document (PRD): Data-Analyst Agent (DAA)

## 1. Tổng quan (Overview)
Data-Analyst Agent (DAA) là hệ thống phân tích và chuẩn hóa hiệu suất nội dung, giúp loại bỏ các yếu tố ngoại cảnh (nhiễu từ xu hướng) để tìm ra chất lượng cốt lõi của bài đăng. Hệ thống sử dụng kỹ thuật **Control Variates (Biến kiểm soát)** và cơ chế dự báo **Cold Start** để đánh giá công bằng cho cả nội dung cũ và mới.

## 2. Module Sinh dữ liệu mẫu (Mock Data Engine)
Hệ thống tự động tạo dữ liệu giả lập cho từng bài viết để phục vụ demo và kiểm thử luồng logic:
* **Numerical Mocking**: Sinh các chỉ số tương tác thô ($Reactions, Comments, Shares, Viewers\_75, Impressions$).
* **Semantic Mocking**: Sinh danh sách 50-100 bình luận giả lập với nhiều sắc thái cảm xúc khác nhau.

## 3. Workflow Xử lý dữ liệu Numerical (Trọng tâm)

Quy trình xác định hiệu suất thực chất của bài đăng thông qua 4 bước logic:

### Bước 1: Tính toán Hiệu suất thực tế ($Y$)
DAA tính toán tỷ lệ tương tác dựa trên trọng số hành động của người dùng:
* **Đối với Video**: $Y = \frac{1 \cdot Reactions + 3 \cdot Comments + 5 \cdot Shares + 2 \cdot Viewers\_75}{Impressions}$.
* **Đối với Bài viết**: $Y = \frac{1 \cdot Reactions + 3 \cdot Comments + 5 \cdot Shares}{Impressions}$.

### Bước 2: Xác định Điểm kỳ vọng ($X_i$) - Cơ chế Rẽ nhánh
Hệ thống thực hiện kiểm tra dữ liệu để xác định "điểm chấp" kỳ vọng cho bài đăng:

* **Trường hợp A: Database Lookup (Đã có lịch sử)**:
    * Nếu thể loại nội dung $i$ đã tồn tại trong Database, hệ thống tra cứu và lấy giá trị Trung bình tích lũy ($Cumulative\ Mean$) của thể loại đó làm $X_i$.
* **Trường hợp B: Cold Start Machine Learning (Thể loại mới)**:
    * Nếu thể loại $i$ chưa từng xuất hiện (ví dụ: Lần đầu đăng Reels), hệ thống kích hoạt luồng dự báo:
        1. **Chuẩn hóa**: Sử dụng `StandardScaler` để xử lý các đặc trưng ($Impressions, Engagement\_Rate$).
        2. **Dự báo**: Sử dụng mô hình `SGDRegressor` để ước lượng giá trị kỳ vọng $X_{cur}$.
        3. **Làm mịn (Smoothing)**: Gán tạm $X_i = \frac{X_{cur} + Global\ Average}{2}$ để đảm bảo tính ổn định cho nội dung mới.
* **Trường hợp C: Extreme Cold Start (Lịch sử rỗng hoặc có ít hơn 2 bài đăng)**:
    * Nếu tài khoản/kênh hoàn toàn mới, số lượng bài đăng $N < 2$, dữ liệu sẽ không đủ điều kiện thống kê để tính phương sai hoặc huấn luyện Machine Learning.
    * **Xử lý ngắt mạch (Fallback)**: Hệ thống bypass AI, gán baseline $\mu_X = 0$, hệ số kiểm soát $\theta = 0$, và kỳ vọng $X_i = Y_{cur}$.
    * **Kết quả**: Hiệu suất điều chỉnh rơi về mức hiệu suất tuyệt đối của chính bài đăng đó ($Y_{adj} = Y_{cur}$), chặn việc lỗi gián đoạn do thiếu dữ liệu. Khoảng tin cậy mở rộng chưa đánh giá mức độ rủi ro ($\pm 0$).

### Bước 3: Hiệu chỉnh Hiệu suất thực chất ($Y_{adj}$)
Sử dụng kỹ thuật **Control Variates** để triệt tiêu nhiễu xu hướng:
* **Công thức**: $Y_{adj} = Y - \theta \cdot (X_i - \mu_X)$.
* **Trong đó**:
    * $\theta$: Hệ số kiểm soát (tính bằng Hiệp phương sai / Phương sai lịch sử).
    * $\mu_X$: Trung bình tương tác của toàn bộ trang (Baseline toàn cục).

### Bước 4: Khoảng tin cậy và Ra quyết định
DAA tính toán Khoảng tin cậy ($CI$) 90% để đưa ra lệnh điều phối:
* **Phương sai tối ưu**: $Var(Y_{adj}) = Var(Y_{hist}) \times (1 - \rho^2)$.
* **Trong đó**: $\rho$ là hệ số tương quan Pearson giữa Lịch sử ($Y_{hist}$) và Kỳ vọng ($X_{hist}$).
* **Logic rẽ nhánh**: 
    * Nếu Cận dưới $CI > \mu_X \rightarrow$ **Volume Up**.
    * Nếu Cận trên $CI < \mu_X \rightarrow$ **Volume Down**.

## 4. Động cơ phân tích ngữ nghĩa (Semantic Engine)
* **Xử lý**: Làm sạch bằng **Text normalizer** và nhúng vector vào **Qdrant DB**.
* **Phân cụm**: Dùng **HDBSCAN** để nhóm các ý kiến tương đồng và chọn ra 10 comment tâm cụm.
* **Phân loại**: LLM định danh cụm thành **Good Feature**, **Bad Feature**, hoặc **Neutral**.

## 5. Ma trận Quyết định (Decision Matrix)
Kết hợp tín hiệu Số liệu và Ngữ nghĩa để rẽ nhánh chiến lược cho **Strategist Agent**:

| Tín hiệu Số liệu | Tín hiệu Ngữ nghĩa | Hành động (Action) |
| :--- | :--- | :--- |
| **Volume Up** | Đa số Good Features | **Keep up (Scale)**: Tăng ngân sách |
| **Volume Down** | Đa số Good Features | **Increase tensity**: Đẩy Seeding, giữ tiền |
| **Volume Up** | Có Bad Features rủi ro | **PR Crisis (Kill)**: Dừng chiến dịch ngay lập tức |
| **Volume Down** | Đa số Bad Features | **Minor tweak**: Tinh chỉnh lại nội dung |

## 6. Yêu cầu Công nghệ (Tech Stack)
* **Ngôn ngữ**: Python.
* **Xử lý số liệu**: Polars.
* **Machine Learning**: `scikit-learn` (`SGDRegressor`, `StandardScaler`).
* **Vector DB**: ChromaDB.

## 7. Hướng dẫn sử dụng (Usage in Multi-Agent System)

Có thể cài đặt repository này như một Python Package và sử dụng bên trong các hệ thống Multi-Agent lớn hơn (ví dụ như gọi dưới dạng công cụ của workflow Orchestrator).

### Cài đặt
Cài đặt trực tiếp qua pip trong chế độ lập trình:
```bash
pip install https://github.com/jethzgg/Data-Analyst-Agent
```

### Cách gọi Agent trong code
```python
from aura_data_analyst import DataAnalystAgent

# 1. Khởi tạo Agent (có thể tuỳ chỉnh API Key và mô hình)
agent = DataAnalystAgent(
    api_key="YOUR_GEMINI_API_KEY", # Bỏ trống để sử dụng file .env cục bộ
    sentiment_model="gemini-2.5-flash", # Hỗ trợ mọi model của Gemini
    embedding_model="gemini-embedding-001" 
)

# 2. Xử lý dữ liệu
result = agent.analyze(
    posts_csv_path="test_data/mock_posts.csv",  
    comments_csv_path="test_data/mock_comments.csv"
)

# 3. Sử dụng Output trả về
print(result['decision'])         # Quyết định hành động (Action)
print(result['semantic_signal'])  # Tín hiệu cảm xúc sau khi gom cụm
print(result['confidence_interval']) # Khoảng tin cậy cho bài toán thống kê
```
