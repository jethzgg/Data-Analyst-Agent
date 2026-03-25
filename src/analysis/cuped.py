import polars as pl
import math

class DataAnalystCUPED:
    """
    Module tính toán các chỉ số và phân tích dựa trên CUPED.
    """

    @staticmethod
    def calculate_theta(df_hist: pl.DataFrame, x_col: str, y_col: str) -> float:
        """
        Tính Toán hệ số Theta dựa trên dữ liệu lịch sử (100 bài biết cũ nhất).
        - x_col: Expected_Trend_Score (X)
        - y_col: Actual_Post_Score (Y)
        """
        cov_xy = df_hist.select(pl.cov(x_col, y_col)).item()
        var_x = df_hist.select(pl.var(x_col)).item()

        # Fallback nếu var_x = 0 hoặc thiếu dữ liệu (Theo tài liệu: fix ở 0.3 - 0.4)
        if var_x is None or var_x == 0 or math.isnan(var_x):
            return 0.3 # Điều chỉnh theta thấp hơn ở trường hợp không có dữ liệu
        
        return cov_xy / var_x

    @staticmethod
    def calculate_variance_cuped(df_hist: pl.DataFrame, theta: float, x_col: str, y_col: str) -> float:
        """
        Tính Variance của Y_CUPED.
        Công thức: var(Y) + theta^2 * var(X) - 2 * theta * cov(X,Y)
        """
        var_y = df_hist.select(pl.var(y_col)).item()
        var_x = df_hist.select(pl.var(x_col)).item()
        cov_xy = df_hist.select(pl.cov(x_col, y_col)).item()

        return var_y + (theta ** 2) * var_x - 2 * theta * cov_xy

    @staticmethod
    def calculate_actual_y_cuped(y_current: float, x_current: float, mu_x: float, theta: float) -> float:
        """
        Tính Y thực tế (Y_CUPED) của bài vừa đăng.
        Công thức: Y thực tế = Y_cur - theta * (X_cur - mu_X)
        - y_current: Kết quả thực tế bài vừa đăng
        - x_current: Mức độ trung bình của Trend tính đến hôm nay
        - mu_x: Trung bình tương tác của toàn bộ Fanpage từ trước đến nay
        - theta: Tỉ lệ tin cậy (covariance / variance)
        """
        return y_current - theta * (x_current - mu_x)

    @staticmethod
    def calculate_video_metric(viewers_75: int, impressions: int) -> float:
        """
        Metric cho Video: Tỷ lệ giữ chân 75%
        """
        if impressions == 0:
            return 0.0
        return viewers_75 / impressions

    @staticmethod
    def calculate_post_metric(reactions: int, comments: int, shares: int, impressions: int) -> float:
        """
        Metric cho Bài viết (Post/Image): Tỷ lệ tương tác có trọng số (ERw).
        """
        if impressions == 0:
            return 0.0
        return (1 * reactions + 3 * comments + 5 * shares) / impressions
