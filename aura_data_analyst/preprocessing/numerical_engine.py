import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import numpy as np

class NumericalEngine:
    """Numerical Engine"""
    
    @staticmethod
    def calculate_actual_score(df: pl.DataFrame) -> pl.DataFrame:
        # 3.1. Calculate Actual Score (Y) - Weighted Average Rate
        # Video weight sum: 1 + 3 + 5 + 2 = 11
        # Other weight sum: 1 + 3 + 5 = 9
        df = df.with_columns(
            pl.when(pl.col("format") == "Video")
            .then((1 * pl.col("reactions") + 3 * pl.col("comments") + 5 * pl.col("shares") + 2 * pl.col("viewers_75")) / (pl.col("impressions") * 11))
            .otherwise((1 * pl.col("reactions") + 3 * pl.col("comments") + 5 * pl.col("shares")) / (pl.col("impressions") * 9))
            .alias("Y")
        )
        return df

    @staticmethod
    def train_historical_model(df_hist: pl.DataFrame):
        # Prevent crash if historical data is extremely small or empty
        if len(df_hist) < 2:
            print("[NumericalEngine] Warning: Not enough historical data. Bypassing ML initialization.")
            df_hist = NumericalEngine.calculate_actual_score(df_hist)
            if "X_hist" not in df_hist.columns:
                df_hist = df_hist.with_columns(pl.lit(0.0).alias("X_hist"))
            if "mu_X" not in df_hist.columns:
                df_hist = df_hist.with_columns(pl.lit(0.0).alias("mu_X"))
            if "Xi" not in df_hist.columns:
                df_hist = df_hist.with_columns(pl.lit(0.0).alias("Xi"))
            format_means = pl.DataFrame({"format": [], "Xi": []})
            return df_hist, format_means, 0.0, None, None

        # Calculate Y for history
        df_hist = NumericalEngine.calculate_actual_score(df_hist)
        
        # Features
        df_hist = df_hist.with_columns(
            ((pl.col("reactions") + pl.col("comments") + pl.col("shares")) / pl.col("impressions")).alias("engagement_rate")
        )
        X_data = df_hist.select(["impressions", "engagement_rate"]).to_numpy()
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        # Train ML
        y_dummy = df_hist.select("Y").to_numpy().ravel()
        model = SGDRegressor(max_iter=1000, tol=1e-3)
        model.fit(X_scaled, y_dummy)
        
        # Predict X_hist
        predicted_X = model.predict(X_scaled)
        df_hist = df_hist.with_columns(pl.Series("X_hist", predicted_X))
        
        # Format Means (Xi for lookup) & mu_X
        # Xi: trung bình cumulative score của từng thể loại i
        format_means = df_hist.group_by("format").agg(pl.col("X_hist").mean().alias("Xi"))
        
        # mu_X: trung bình score của tất cả thể loại trong lịch sử
        mu_X = df_hist["X_hist"].mean()
        df_hist = df_hist.with_columns(pl.lit(mu_X).alias("mu_X"))
        df_hist = df_hist.join(format_means, on="format", how="left")
        
        return df_hist, format_means, mu_X, model, scaler
        
    @staticmethod
    def evaluate_current_post(df_cur: pl.DataFrame, format_means: pl.DataFrame, mu_X: float, model: SGDRegressor, scaler: StandardScaler):
        df_cur = NumericalEngine.calculate_actual_score(df_cur)
        
        df_cur = df_cur.with_columns(
            ((pl.col("reactions") + pl.col("comments") + pl.col("shares")) / pl.col("impressions")).alias("engagement_rate")
        )
        
        # Process each row (assuming mostly 1 row for current post)
        X_cur = []
        for row in df_cur.iter_rows(named=True):
            f_format = row["format"]
            
            # Check if format exists in history (Lookup)
            match_format = format_means.filter(pl.col("format") == f_format)
            
            if len(match_format) > 0:
                # Lookup
                xi_val = match_format["Xi"][0]
                print(f"[NumericalEngine] Format '{f_format}' Found in history. Lookup Xi: {xi_val:.4f}")
            elif model is not None and scaler is not None:
                # Cold Start (Predict using ML)
                cur_features = np.array([[row["impressions"], row["engagement_rate"]]])
                cur_scaled = scaler.transform(cur_features)
                xi_val = model.predict(cur_scaled)[0]
                print(f"[NumericalEngine] COLD START for format '{f_format}'. Predicted Xi via ML: {xi_val:.4f}")
            else:
                # Extreme Cold Start: No ML model available due to lack of historical data
                xi_val = row["Y"] # Fallback to using its own Y as baseline
                print(f"[NumericalEngine] EXTREME COLD START. No history available. Fallback Xi: {xi_val:.4f}")
            
            X_cur.append(xi_val)
            
        df_cur = df_cur.with_columns(
            pl.Series("Xi", X_cur),
            pl.lit(mu_X).alias("mu_X")
        )
        
        return df_cur
