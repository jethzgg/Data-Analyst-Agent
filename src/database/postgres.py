class PostgresDB:
    # Bỏ qua logic kết nối thật để test offline
    def __init__(self, dsn):
        self.dsn = dsn
        
    def save_numerical_data(self, df, table_name):
        pass
        
    def get_historical_mu(self, df):
        if len(df) > 0:
            return df["X"].mean()
        return 0.0
