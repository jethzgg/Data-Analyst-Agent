# Product Requirement Document (PRD): Data-Analyst Agent (DAA)

## 1. Tổng quan (Overview)
Data-Analyst Agent (DAA) là thành phần chịu trách nhiệm xử lý, chuẩn hóa và đánh giá hiệu suất của từng bài đăng độc lập. Thay vì sử dụng dữ liệu thô (raw data) thường bị nhiễu bởi các yếu tố ngoại cảnh, DAA áp dụng kỹ thuật **Control Variates (Biến kiểm soát)** kết hợp với **Machine Learning** để xác định giá trị chất lượng thực chất (core quality) của nội dung.

## 2. Module Sinh dữ liệu mẫu (Mock Data Engine)
Hệ thống tích hợp trình sinh dữ liệu giả lập cho từng bài viết để phục vụ demo và kiểm thử luồng logic mà không cần kết nối API thực tế:
* **Numerical Mocking**: Tự động gán bộ chỉ số tương tác thô ($Reactions, Comments, Shares, Viewers\_75, Impressions$) cho một bài đăng cụ thể.
* **Semantic Mocking**: Sinh danh sách 50-100 bình luận giả lập với đa dạng sắc thái từ tích cực, tiêu cực đến các từ khóa rủi ro truyền thông.

## 3. Động cơ xử lý dữ liệu số (Numerical Engine)
Đây là lõi tính toán của Agent, tập trung vào việc khử nhiễu và dự báo kỳ vọng.

### 3.1. Tính toán Hiệu suất thực tế ($Y$)
Tỷ lệ tương tác được tính toán dựa trên trọng số ưu tiên hành động của người dùng đối với từng loại định dạng:
* **Đối với Video**: $Y = \frac{1 \cdot Reactions + 3 \cdot Comments + 5 \cdot Shares + 2 \cdot Viewers\_75}{Impressions}$.
* **Đối với Bài viết (Post/Image)**: $Y = \frac{1 \cdot Reactions + 3 \cdot Comments + 5 \cdot Shares}{Impressions}$.

### 3.2. Dự báo Điểm kỳ vọng của thể loại ($X_i$)
Hệ thống xử lý bài toán **Cold Start** (nội dung mới) bằng phương pháp học máy tăng cường:
* **Chuẩn hóa**: Sử dụng `StandardScaler` để đưa các đặc trưng về cùng một phân phối.
* **Dự báo (ML)**: Sử dụng mô hình `SGDRegressor` để ước lượng $X_i$ dựa trên quy mô hiển thị ($Impressions$) và tỷ lệ tương tác thô ($Engagement\_Rate$).
* **Làm mịn (Smoothing)**: Giá trị $X_i$ cuối cùng là trung bình giữa giá trị dự báo và trung bình toàn cục của Page: $X_i = \frac{Predicted\_X + Global\_Average}{2}$.

### 3.3. Hiệu chỉnh bằng Biến kiểm soát (Control Variates)
* **Hệ số kiểm soát ($\theta$)**: Tính toán từ dữ liệu lịch sử: $\theta = \frac{Cov(X, Y)}{Var(X)}$ (Nếu thiếu dữ liệu, fix cứng ở mức 0.35).
* **Hiệu suất thực chất ($Y_{adj}$)**: Triệt tiêu "điểm chấp" của xu hướng: $Y_{adj} = Y - \theta \cdot (X_i - \mu_X)$ với $\mu_X$ là trung bình tương tác toàn page.
* **Khoảng tin cậy (CI)**: Tính toán Khoảng tin cậy 90% dựa trên phương sai đã tối ưu: $Var(Y_{adj}) = Var(Y) + \theta^2 \cdot Var(X) - 2\theta \cdot Cov(X, Y)$.

## 4. Động cơ phân tích ngữ nghĩa (Semantic Engine)
* **Xử lý văn bản**: Làm sạch bằng **Text normalizer** và lưu trữ dưới dạng vector trong **Qdrant DB**.
* **Phân cụm**: Sử dụng thuật toán **HDBSCAN** để nhóm các bình luận tương đồng.
* **Phân loại cảm xúc**: LLM trích xuất 10 bình luận tâm cụm để xác định các đặc tính: **Good Feature**, **Bad Feature**, hoặc **Neutral**.

## 5. Ma trận Quyết định Tổng hợp (Final Decision)
DAA kết hợp tín hiệu Số liệu và Ngữ nghĩa để rẽ nhánh chiến lược cho **Strategist Agent**:

| Tín hiệu Số liệu ($CI$ vs $\mu_X$) | Tín hiệu Ngữ nghĩa | Hành động (Action) |
| :--- | :--- | :--- |
| **Cận dưới $CI > \mu_X$** | Đa số Good Features | **Keep up (Scale)**: Tăng ngân sách |
| **Cận trên $CI < \mu_X$** | Đa số Good Features | **Increase tensity**: Đẩy Seeding |
| **Cận dưới $CI > \mu_X$** | Xuất hiện Bad Features rủi ro | **PR Crisis (Kill)**: Dừng chiến dịch |
| **Cận trên $CI < \mu_X$** | Đa số Bad Features | **Minor tweak**: Sửa nội dung |

## 6. Yêu cầu Công nghệ (Tech Stack)
* **Ngôn ngữ**: Python.
* **Xử lý dữ liệu**: Polars.
* **Machine Learning**: `scikit-learn` (`SGDRegressor`, `StandardScaler`).
* **Vector Database**: Qdrant.
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
result = agent.process_pipeline(
    posts_csv_path="test_data/mock_posts.csv",  
    comments_csv_path="test_data/mock_comments.csv"
)

# 3. Sử dụng Output trả về
print(result['decision'])         # Quyết định hành động (Action)
print(result['semantic_signal'])  # Tín hiệu cảm xúc sau khi gom cụm
print(result['confidence_interval']) # Khoảng tin cậy cho bài toán thống kê
```
