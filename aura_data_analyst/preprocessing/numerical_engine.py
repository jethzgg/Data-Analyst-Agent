import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import numpy as np

class NumericalEngine:
    """Numerical Engine"""
    
    @staticmethod
    def calculate_actual_score(df: pl.DataFrame) -> pl.DataFrame:
        # 3.1. Calculate Actual Score (Y)
        df = df.with_columns(
            pl.when(pl.col("format") == "Video")
            .then((1 * pl.col("reactions") + 3 * pl.col("comments") + 5 * pl.col("shares") + 2 * pl.col("viewers_75")) / pl.col("impressions"))
            .otherwise((1 * pl.col("reactions") + 3 * pl.col("comments") + 5 * pl.col("shares")) / pl.col("impressions"))
            .alias("Y")
        )
        return df

    @staticmethod
    def predict_expected_score(df: pl.DataFrame) -> pl.DataFrame:
        # 3.2. Predict Expected Score (Xi)
        # Engagement_Rate = (reactions + comments + shares) / impressions
        df = df.with_columns(
            ((pl.col("reactions") + pl.col("comments") + pl.col("shares")) / pl.col("impressions")).alias("engagement_rate")
        )
        
        # Prepare features
        X_data = df.select(["impressions", "engagement_rate"]).to_numpy()
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        # Prediction (ML) - Using dummy target for training since this is Cold Start proxy
        y_dummy = df.select("Y").to_numpy().ravel()
        model = SGDRegressor(max_iter=1000, tol=1e-3)
        model.fit(X_scaled, y_dummy)
        
        predicted_X = model.predict(X_scaled)
        global_average = df["Y"].mean()
        
        # Smoothing
        smoothed_X = (predicted_X + global_average) / 2
        
        df = df.with_columns(pl.Series("X", smoothed_X))
        return df
