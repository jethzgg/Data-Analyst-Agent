import polars as pl
import numpy as np

class ControlVariates:
    """Hiệu chỉnh bằng Biến kiểm soát (Control Variates)"""
    
    @staticmethod
    def calculate_theta(df: pl.DataFrame, x_col="X", y_col="Y") -> float:
        if len(df) < 2:
            return 0.35 # Fix cứng nếu thiếu dữ liệu
        
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        
        cov_matrix = np.cov(x, y)
        if cov_matrix[0,0] == 0:
            return 0.35
            
        theta = cov_matrix[0, 1] / cov_matrix[0, 0]
        return theta if not np.isnan(theta) else 0.35

    @staticmethod
    def apply_cuped(df: pl.DataFrame, theta: float, mu_x: float) -> pl.DataFrame:
        # Y_adj = Y - theta * (X_i - mu_X)
        df = df.with_columns(
            (pl.col("Y") - theta * (pl.col("X") - mu_x)).alias("Y_adj")
        )
        return df

    @staticmethod
    def calculate_ci(df: pl.DataFrame, theta: float) -> tuple:
        var_y = df["Y"].var() if len(df) > 1 else 0
        var_x = df["X"].var() if len(df) > 1 else 0
        cov_xy = np.cov(df["X"].to_numpy(), df["Y"].to_numpy())[0, 1] if len(df) > 1 else 0
        
        # Var(Y_adj) = Var(Y) + theta^2 * Var(X) - 2 * theta * Cov(X, Y)
        var_y_adj = var_y + (theta**2 * var_x) - (2 * theta * cov_xy)
        
        # Approximate SE and CI
        n = len(df)
        se = np.sqrt(max(var_y_adj, 0) / n) if n > 0 else 0
        ci_90 = 1.645 * se # 90% CI
        
        return var_y_adj, ci_90
