import polars as pl
import numpy as np
from scipy.stats import norm

class ControlVariates:
    """Control Variates Adjustment"""
    
    @staticmethod
    def calculate_theta(df: pl.DataFrame, x_col="X_hist", y_col="Y") -> float:
        if len(df) < 2:
            return 0.0 # No variance reduction possible tracking
        
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        
        cov_matrix = np.cov(x, y)
        if cov_matrix[0,0] == 0:
            return 0.35
            
        theta = cov_matrix[0, 1] / cov_matrix[0, 0]
        return theta if not np.isnan(theta) else 0.35

    @staticmethod
    def apply_control_variates(df: pl.DataFrame, theta: float, mu_x: float, x_col="Xi") -> pl.DataFrame:
        # Y_adj = Y - theta * (X_i - mu_X)
        df = df.with_columns(
            (pl.col("Y") - theta * (pl.col(x_col) - mu_x)).alias("Y_adj")
        )
        return df

    @staticmethod
    def calculate_ci(df: pl.DataFrame, theta: float, x_col="X_hist", ci_level: float = 0.95) -> tuple:
        var_y = df["Y"].var() if len(df) > 1 else 0
        
        # Calculate Pearson correlation coefficient rho
        if len(df) > 1:
            rho = np.corrcoef(df[x_col].to_numpy(), df["Y"].to_numpy())[0, 1]
            if np.isnan(rho):
                rho = 0
        else:
            rho = 0
            
        # Var(Y_adj) = Var(Y_hist) * (1 - rho^2)
        var_y_adj = var_y * (1 - rho**2)
        
        # Approximate SE and CI
        n = len(df)
        se = np.sqrt(max(var_y_adj, 0) / n) if n > 0 else 0
        
        # Calculate Z-score based on the desired Confidence Level
        z_score = norm.ppf((1 + ci_level) / 2)
        ci_margin = z_score * se 
        
        return var_y_adj, ci_margin
