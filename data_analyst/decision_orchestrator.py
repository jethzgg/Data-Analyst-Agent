import polars as pl
from data_analyst.collection.mock_data_engine import MockDataEngine
from data_analyst.preprocessing.numerical_engine import NumericalEngine
from data_analyst.analysis.control_variates import ControlVariates
from data_analyst.database.postgres import PostgresDB

class FinalDecisionMatrix:
    def __init__(self):
        self.pg_db = PostgresDB("postgres://mock")
        
    def analyze(self, posts_csv_path=None, ci_level=0.95):
        print("\n1. MOCK DATA ENGINE")
        df_posts = MockDataEngine.generate_posts(csv_path=posts_csv_path)
        
        # Simulate history & current post split.
        df_hist = df_posts[:-1]
        latest_post = df_posts[-1:]
        
        print("\n2. NUMERICAL ENGINE")
        df_hist, format_means, mu_x, model, scaler = NumericalEngine.train_historical_model(df_hist)
        
        # Demo Cold Start Feature by temporarily modifying the latest post format to an unknown format
        # Uncomment the line below to force a Cold Start for testing
        # latest_post = latest_post.with_columns(pl.lit("New_Unknown_Format").alias("format"))
        
        print("\nNUMERICAL ENGINE")
        # Check if format exists or predict.
        latest_post = NumericalEngine.evaluate_current_post(latest_post, format_means, mu_x, model, scaler)
        
        print("\n3. CONTROL VARIATES ANALYSIS")
        theta = ControlVariates.calculate_theta(df_hist, "X_hist", "Y")
        latest_post_adj = ControlVariates.apply_control_variates(latest_post, theta, mu_x)
        
        y_adj = latest_post_adj["Y_adj"][0]
        
        var_y_adj, ci_margin = ControlVariates.calculate_ci(df_hist, theta, ci_level=ci_level)
        
        lower_bound = y_adj - ci_margin
        upper_bound = y_adj + ci_margin
        
        print(f"Y_adj: {y_adj:.4f}, mu_X: {mu_x:.4f}")
        print(f"Confidence Interval {int(ci_level * 100)}%: [{lower_bound:.4f} - {upper_bound:.4f}]")
        
        print("\n4. FINAL DECISION")
        if lower_bound > mu_x:
            decision = "Volume Up: Content performing above average"
        elif upper_bound < mu_x:
            decision = "Volume Down: Content performing below average"
        else:
            decision = "Inconclusive: Not enough statistical evidence"
            
        print(f"DECISION: {decision}")
        
        return {
            "decision": decision,
            "y_adj": y_adj,
            "mu_x": mu_x,
            "confidence_interval": [lower_bound, upper_bound],
        }

if __name__ == "__main__":
    A = FinalDecisionMatrix()
    A.analyze()
