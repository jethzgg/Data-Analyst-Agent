import polars as pl
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    @staticmethod
    def calculate_actual_score(df: pl.DataFrame) -> pl.DataFrame:
        """Tính Actual_Post_Score theo công thức trong file cuped."""
        # Bước 1: Tính Y (Actual Score) 
        df = df.with_columns(
            pl.when(pl.col("type") == "Video")
            .then(
                (pl.col('reactions') * 1 + pl.col('comments') * 3 + pl.col('shares') * 5 + pl.col('viewers_75') * 2)
                / pl.col('impressions')
            )
            .otherwise(
                (pl.col('reactions') * 1 + pl.col('comments') * 3 + pl.col('shares') * 5)
                / pl.col('impressions')
            ).alias('actual_score')
        )
        
        # Bước 2: Phân rã trung bình tích luỹ theo từng loại
        df = df.with_columns(
            pl.col("actual_score")
            .shift(1)
            .cumulative_eval(pl.element().mean())
            .over("type")
            .alias("computed_category_avg")
        )
        
        # Bước 3: Xử lý chuyên sâu Cold Start với Hồi quy phi tuyến (Học tăng cường)
        records = df.to_dicts()
        
        # Công cụ cho Học Tăng Cường (Incremental Learning)
        scaler = StandardScaler()
        model = SGDRegressor(penalty='l2', random_state=42)
        
        historical_scores = [] # Dùng để tính Global Average (Làm mịn)
        
        for row in records:
            # Feature Engineering: Thêm tỷ lệ tương tác (engagement_rate)
            eng_rate = (row['reactions'] + row['comments'] + row['shares']) / row['impressions'] if row['impressions'] > 0 else 0
            current_features = np.array([[row["impressions"], eng_rate]])
            
            if row["computed_category_avg"] is None:
                if len(historical_scores) < 2:
                    row["category_avg_score"] = 0.05
                else:
                    # Scaler transform dựa trên dữ liệu đã parse
                    scaled_features = scaler.transform(current_features)
                    predicted_x = model.predict(scaled_features)[0]
                    
                    # Làm mịn dự đoán (Prediction Smoothing)
                    global_avg = sum(historical_scores) / len(historical_scores)
                    smoothed_x = (predicted_x + global_avg) / 2
                    
                    row["category_avg_score"] = smoothed_x
                    print(f"\n[AI ML] SGDRegressor (Học tăng cường) kích hoạt cho Thể loại mới: '{row['type']}' (ID: {row['post_id']})")
                    print(f"-> Học từ {len(historical_scores)} bài cũ.")
                    print(f"-> Raw Predict: {predicted_x:.4f} | Global Avg: {global_avg:.4f} => Smoothed X_i: {smoothed_x:.4f}")
            else:
                row["category_avg_score"] = row["computed_category_avg"]
            
            # Học Tăng Cường (Online Learning) sau khi đã xử lý xong row hiện tại
            scaler.partial_fit(current_features)
            scaled_for_fit = scaler.transform(current_features)
            target = np.array([row["actual_score"]])
            model.partial_fit(scaled_for_fit, target)
            
            historical_scores.append(row["actual_score"])
            
        df_final = pl.DataFrame(records).drop("computed_category_avg", strict=False)
        return df_final