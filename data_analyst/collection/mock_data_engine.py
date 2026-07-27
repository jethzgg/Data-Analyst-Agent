import polars as pl
import os

# Get absolute path to the project root (Data-Analyst-Agent folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MockDataEngine:
    """Mock Data Engine Module"""
    
    @staticmethod
    def generate_posts(csv_path=None) -> pl.DataFrame:
        """Loads mocked posts from a static CSV file instead of generating them randomly."""
        if not csv_path:
            csv_path = os.path.join(BASE_DIR, "test_data", "mock_posts.csv")
        return pl.read_csv(csv_path)

