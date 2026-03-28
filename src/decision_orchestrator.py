import polars as pl
from src.collection.mock_data_engine import MockDataEngine
from src.preprocessing.numerical_engine import NumericalEngine
from src.analysis.control_variates import ControlVariates
from src.database.postgres import PostgresDB
from src.analysis.semantic.semantic_engine import SemanticEngine

class FinalDecisionMatrix:
    def __init__(self):
        self.pg_db = PostgresDB("postgres://mock")
        self.semantic_engine = SemanticEngine()
        
    def process_pipeline(self):
        print("\n=== 1. MOCK DATA ENGINE ===")
        df_posts = MockDataEngine.generate_posts(100)
        comments = MockDataEngine.generate_comments()
        
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

if __name__ == "__main__":
    A = AgentOrchestrator()
    A.process_pipeline()
