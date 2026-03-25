import polars as pl

class PostgresDB:
    """Mock Connection cho PostgreSQL"""
    def __init__(self, connection_string: str):
        self.conn = connection_string
        print(f"[Postgres] Đã kết nối DB giả lập {self.conn}")

    def save_numerical_data(self, df: pl.DataFrame, table_name: str):
        """Lưu trữ Data lên DB (Giả lập)"""
        print(f"[Postgres] Đã lưu {df.shape[0]} dòng vào bảng {table_name}")

    def get_historical_mu(self, df_posts: pl.DataFrame = None) -> float:
        """
        Lấy điểm mu_x trung bình tương tác.
        Theo thiết kế, mu_X là trung bình của các 'thể loại'. Nghĩa là ta tính trung bình
        của tất cả các X_i (Category Avg) đang có, để lấy mức base mặt bằng chung
        """
        if df_posts is not None and "category_avg_score" in df_posts.columns:
            # Lấy danh sách các thể loại và mức trung bình đại diện của chúng
            # Bước 1: Chỉ lấy các giá trị trung bình hợp lệ
            valid_scores = df_posts.drop_nulls("category_avg_score")
            if len(valid_scores) > 0:
                # Bước 2: Group by theo type để lấy đúng trung bình của từng thể loại đó (1 giá trị cho Video, 1 cho Bài viết)
                grouped_means = valid_scores.group_by("type").agg(
                    pl.col("category_avg_score").last().alias("latest_cat_avg")
                )
                # Bước 3: Tính mu_X = Trung bình cộng của các trung bình thể loại
                return grouped_means.select(pl.col("latest_cat_avg").mean()).item()
        
        return 0.05  # Fallback: Ví dụ 5% trung bình toàn page
