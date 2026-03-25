import polars as pl
import os

class DataLoader:
    """Generic CSV dataloader for local test data."""
    def __init__(self, file_path: str = "test_data/mock_posts.csv"):
        self.file_path = file_path

    def load_data(self) -> pl.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Mock data not found at: {self.file_path}")
        print(f"Loading data from {self.file_path}...")
        df = pl.read_csv(self.file_path)
        print(f"Loaded {len(df)} records.")
        return df

