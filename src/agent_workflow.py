import polars as pl
from src.collection.data_loader import DataLoader
from src.preprocessing.data_processor import DataProcessor
from src.analysis.cuped import DataAnalystCUPED
from src.database.postgres import PostgresDB

class AgentOrchestrator:
    def __init__(self):
        self.pg_db = PostgresDB("postgres://mock")
        self.data_loader = DataLoader(file_path="test_data/mock_posts.csv")
        
    def process_pipeline(self):
        print("\n=== 1. DATA COLLECTION ===")
        df_posts = self.data_loader.load_data()
        
        print("\n=== 2. NUMERICAL PREPROCESSING ===")
        df_posts = DataProcessor.calculate_actual_score(df_posts)
        self.pg_db.save_numerical_data(df_posts, "posts_metric")
        
        print("\n=== 3. CUPED ANALYSIS ===")
        # Phân tách rõ ràng: Dữ liệu lịch sử (Tất cả trừ bài cuối) VÀ Bài viết hiện tại (Bài cuối cùng)
        df_hist = df_posts.slice(0, len(df_posts) - 1)
        latest_post = df_posts.row(-1, named=True)
        
        theta = DataAnalystCUPED.calculate_theta(
            df_hist, x_col="category_avg_score", y_col="actual_score"
        )
        mu_x = self.pg_db.get_historical_mu(df_hist)
        
        print(f"\n=== HIỆU SUẤT LỊCH SỬ (Pre-Experiment) ===")
        print(f"Global Expected Mean (\u03bc_X): {mu_x}")
        print(f"Theta (Tính từ History) : {theta:.4f}")
        
        y_cur = latest_post["actual_score"]
        x_cur = latest_post["category_avg_score"]
        n_impressions = latest_post["impressions"]
        
        y_cuped = DataAnalystCUPED.calculate_actual_y_cuped(y_cur, x_cur, mu_x, theta)
        
        print(f"\n[Latest Post Output - ID {latest_post['post_id']}]")
        print(f"- X_{{cur}} (Category Avg): {x_cur:.4f}")
        print(f"- Y_{{cur}} (Actual Raw)  : {y_cur:.4f}")
        print(f"=> Y_CUPED (Hiệu suất): {y_cuped:.4f}")
        
        # Tính phương sai và khoảng tin cậy 90% (Z = 1.645)
        import math
        var_cuped = DataAnalystCUPED.calculate_variance_cuped(
            df_hist, theta, x_col="category_avg_score", y_col="actual_score"
        )
        # Sử dụng N = impressions hiện tại để đánh giá Statistical Significance
        std_err = math.sqrt(var_cuped / n_impressions) if var_cuped > 0 and n_impressions > 0 else 0
        margin_of_error = 1.645 * std_err
        
        lower_bound = y_cuped - margin_of_error
        upper_bound = y_cuped + margin_of_error
        
        print("\n=== 4. THỐNG KÊ & KHOẢNG TIN CẬY 90% ===")
        print(f"Variance (Y_CUPED) : {var_cuped:.6f}")
        print(f"Standard Error     : {std_err:.6f}")
        print(f"Confidence Interval: [{lower_bound:.4f} , {upper_bound:.4f}]")
        print(f"Ngưỡng so sánh \u03bc_X: {mu_x}")
        
        print("\n=== 5. QUYẾT ĐỊNH (DECISION) ===")
        if lower_bound > mu_x:
            print("=> Cận dưới > \u03bc_X")
            print("-> Indicator: VOLUME UP (Đẩy mạnh phân phối / Ngân sách quảng cáo)")
        elif upper_bound < mu_x:
            print("=> Cận trên < \u03bc_X")
            print("-> Indicator: VOLUME DOWN (Giảm phân phối / Ngừng chiến dịch)")
        else:
            print(f"=> \u03bc_X ({mu_x}) nằm trong khoảng [{lower_bound:.4f} - {upper_bound:.4f}]")
            print("-> Indicator: KHÔNG ĐỦ Ý NGHĨA THỐNG KÊ (Tiếp tục theo dõi / Giữ nguyên hiện trạng)")
