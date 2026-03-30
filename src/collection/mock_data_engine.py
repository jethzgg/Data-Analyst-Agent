import polars as pl

class MockDataEngine:
    """Mock Data Engine Module"""
    
    @staticmethod
    def generate_posts(n_posts=None) -> pl.DataFrame:
        """Loads mocked posts from a static CSV file instead of generating them randomly."""
        return pl.read_csv("test_data/mock_posts.csv")

    @staticmethod
    def generate_comments() -> list:
        """Loads mocked comments from a static CSV file and returns them as a list of dicts."""
        df = pl.read_csv("test_data/mock_comments.csv")
        return df.to_dicts()
