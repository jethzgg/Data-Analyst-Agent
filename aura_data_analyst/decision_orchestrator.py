import polars as pl
from aura_data_analyst.collection.mock_data_engine import MockDataEngine
from aura_data_analyst.preprocessing.numerical_engine import NumericalEngine
from aura_data_analyst.analysis.control_variates import ControlVariates
from aura_data_analyst.database.postgres import PostgresDB
from aura_data_analyst.analysis.semantic.semantic_engine import SemanticEngine

class FinalDecisionMatrix:
    def __init__(self, api_key: str = None, sentiment_model: str = 'gemini-2.5-flash', embedding_model: str = 'gemini-embedding-001'):
        self.pg_db = PostgresDB("postgres://mock")
        self.semantic_engine = SemanticEngine()
        
    def analyze(self, posts_csv_path=None, comments_csv_path=None):
        print("\n=== 1. MOCK DATA ENGINE ===")
        df_posts = MockDataEngine.generate_posts(100, csv_path=posts_csv_path)
        comments = MockDataEngine.generate_comments(csv_path=comments_csv_path)
        
        # Simulate history & current post split.
        df_hist = df_posts[:-1]
        latest_post = df_posts[-1:]
        
        print("\n=== 2. NUMERICAL ENGINE (Historical Training & Lookup Generation) ===")
        df_hist, format_means, mu_x, model, scaler = NumericalEngine.train_historical_model(df_hist)
        
        # Demo Cold Start Feature by temporarily modifying the latest post format to an unknown format
        # Uncomment the line below to force a Cold Start for testing
        # latest_post = latest_post.with_columns(pl.lit("New_Unknown_Format").alias("format"))
        
        print("\n=== NUMERICAL ENGINE (Evaluate Current Post - Cold Start/Lookup) ===")
        # Check if format exists or predict.
        latest_post = NumericalEngine.evaluate_current_post(latest_post, format_means, mu_x, model, scaler)
        
        print("\n=== 3. CONTROL VARIATES ANALYSIS ===")
        theta = ControlVariates.calculate_theta(df_hist, "X_hist", "Y")
        latest_post_adj = ControlVariates.apply_control_variates(latest_post, theta, mu_x)
        
        y_adj = latest_post_adj["Y_adj"][0]
        
        var_y_adj, ci_margin = ControlVariates.calculate_ci(df_hist, theta)
        
        lower_bound = y_adj - ci_margin
        upper_bound = y_adj + ci_margin
        
        print(f"Y_adj: {y_adj:.4f}, mu_X: {mu_x:.4f}")
        print(f"Confidence Interval 90%: [{lower_bound:.4f} - {upper_bound:.4f}]")
        
        print("\n=== 4. SEMANTIC ENGINE ===")
        try:
            semantic_signal = self.semantic_engine.analyze_social_sentiment(df_posts, comments)
            reason = semantic_signal.get("reason", "Neutral")
        except AttributeError:
            # Fallback for old version
            try:
                semantic_signal = self.semantic_engine.analyze_comments(comments)
                reason = semantic_signal
            except Exception:
                reason = "Neutral"

        print(f"Semantic Feature: {reason}")
        
        print("\n=== 5. FINAL DECISION ===")
        decision = "Wait and see (Not in fixed rule)"
        if lower_bound > mu_x and reason in ["Good Feature", "Neutral"]:
            decision = "Keep up (Scale): Increase budget"
        elif upper_bound < mu_x and reason in ["Good Feature", "Neutral"]:
            decision = "Increase tensity: Push seeding"
        elif lower_bound > mu_x and reason == "Bad Feature":
            decision = "PR Crisis (Kill): Stop campaign"
        elif upper_bound < mu_x and reason == "Bad Feature":
            decision = "Minor tweak: Edit content"
            
        print(f"-> DECISION: {decision}")
        
        return {
            "decision": decision,
            "y_adj": y_adj,
            "mu_x": mu_x,
            "confidence_interval": [lower_bound, upper_bound],
            "semantic_signal": reason
        }

if __name__ == "__main__":
    A = FinalDecisionMatrix()
    A.analyze()
