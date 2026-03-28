import polars as pl
import numpy as np
import random

class MockDataEngine:
    """Module Sinh dữ liệu mẫu (Mock Data Engine)"""
    @staticmethod
    def generate_posts(n_posts=100) -> pl.DataFrame:
        data = []
        for i in range(n_posts):
            format_type = random.choice(["Video", "Post", "Image"])
            impressions = int(np.random.normal(50000, 15000))
            if impressions < 1000: impressions = 1000
            
            reactions = int(impressions * random.uniform(0.01, 0.05))
            comments = int(impressions * random.uniform(0.005, 0.02))
            shares = int(impressions * random.uniform(0.001, 0.01))
            viewers_75 = int(impressions * random.uniform(0.1, 0.4)) if format_type == "Video" else 0
            
            data.append({
                "post_id": f"post_{i}",
                "format": format_type,
                "impressions": impressions,
                "reactions": reactions,
                "comments": comments,
                "shares": shares,
                "viewers_75": viewers_75
            })
        return pl.DataFrame(data)

    @staticmethod
    def generate_comments() -> list:
        # Semantic Mocking
        sentiments = ["Tích cực", "Tiêu cực", "Trung tính", "Từ khóa rủi ro"]
        comments = []
        for _ in range(50):
            comments.append({"text": "Mock comment", "sentiment_hint": random.choice(sentiments)})
        return comments
