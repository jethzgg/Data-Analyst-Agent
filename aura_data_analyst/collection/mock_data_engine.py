import polars as pl
import os

# Get absolute path to the project root (Data-Analyst-Agent folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MockDataEngine:
    """Mock Data Engine Module"""
    
    @staticmethod
    def generate_posts(n_posts=None, csv_path=None) -> pl.DataFrame:
        """Loads mocked posts from a static CSV file instead of generating them randomly."""
        if not csv_path:
            csv_path = os.path.join(BASE_DIR, "test_data", "mock_posts.csv")
        return pl.read_csv(csv_path)

    @staticmethod
    def generate_comments(csv_path=None) -> list:
        """Loads mocked comments from a static CSV file and returns them as a list of dicts."""
        if not csv_path:
            csv_path = os.path.join(BASE_DIR, "test_data", "mock_comments.csv")
        df = pl.read_csv(csv_path)
        return df.to_dicts()
