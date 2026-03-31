import polars as pl
from aura_data_analyst.collection.mock_data_engine import MockDataEngine
from aura_data_analyst.preprocessing.numerical_engine import NumericalEngine
from aura_data_analyst.analysis.control_variates import ControlVariates
from aura_data_analyst.database.postgres import PostgresDB
from aura_data_analyst.analysis.semantic.semantic_engine import SemanticEngine

class FinalDecisionMatrix:
    def __init__(self, api_key: str = None, sentiment_model: str = 'gemini-2.5-flash', embedding_model: str = 'gemini-embedding-001'):
        self.pg_db = PostgresDB("postgres://mock")
        self.semantic_engine = SemanticEngine(
            api_key=api_key,
            sentiment_model=sentiment_model,
            embedding_model=embedding_model
        )
        
    def process_pipeline(self, posts_csv_path=None, comments_csv_path=None):
        print("\n=== 1. MOCK DATA ENGINE ===")
        df_posts = MockDataEngine.generate_posts(100, csv_path=posts_csv_path)
        comments = MockDataEngine.generate_comments(csv_path=comments_csv_path)
        
        print("\n=== 2. NUMERICAL ENGINE ===")
        # 3.1 and 3.2
        df_posts = NumericalEngine.calculate_actual_score(df_posts)
        df_posts = NumericalEngine.predict_expected_score(df_posts)
        
        print("\n=== 3. CONTROL VARIATES ANALYSIS ===")
        # Simulate last post as current
        df_hist = df_posts[:-1]
        latest_post = df_posts[-1:]
        
        mu_x = df_hist["X"].mean()
        theta = ControlVariates.calculate_theta(df_hist, "X", "Y")
        
        latest_post_adj = ControlVariates.apply_control_variates(latest_post, theta, mu_x)
        y_adj = latest_post_adj["Y_adj"][0]
        
        var_y_adj, ci_margin = ControlVariates.calculate_ci(df_hist, theta)
        
        lower_bound = y_adj - ci_margin
        upper_bound = y_adj + ci_margin
        
        print(f"Y_adj: {y_adj:.4f}, \u03bc_X: {mu_x:.4f}")
        print(f"Confidence Interval 90%: [{lower_bound:.4f} - {upper_bound:.4f}]")
        
        print("\n=== 4. SEMANTIC ENGINE ===")
        semantic_signal = self.semantic_engine.analyze_comments(comments)
        print(f"Semantic Feature: {semantic_signal}")
        
        print("\n=== 5. FINAL DECISION ===")
        # Decision Matrix
        if lower_bound > mu_x and semantic_signal in ["Good Feature", "Neutral"]:
            decision = "Keep up (Scale): Increase budget"
        elif upper_bound < mu_x and semantic_signal in ["Good Feature", "Neutral"]:
            decision = "Increase tensity: Push seeding"
        elif lower_bound > mu_x and semantic_signal == "Bad Feature":
            decision = "PR Crisis (Kill): Stop campaign"
        elif upper_bound < mu_x and semantic_signal == "Bad Feature":
            decision = "Minor tweak: Edit content"
        else:
            decision = "Wait and see (Not in fixed rule)"
            
        print(f"-> DECISION: {decision}")
        
        return {
            "decision": decision,
            "y_adj": y_adj,
            "mu_x": mu_x,
            "confidence_interval": [lower_bound, upper_bound],
            "semantic_signal": semantic_signal
        }

if __name__ == "__main__":
    A = FinalDecisionMatrix()
    A.process_pipeline()
