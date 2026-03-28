import numpy as np

class SemanticEngine:
    """Động cơ phân tích ngữ nghĩa (Semantic Engine)"""
    
    def __init__(self):
        # Placeholder for Qdrant and HDBSCAN
        pass
        
    def analyze_comments(self, comments: list) -> str:
        """
        Mock analysis of comments
        Returns: 'Good Feature', 'Bad Feature', or 'Neutral'
        """
        # Mocking semantic logic based on generated hints
        sentiments = [c['sentiment_hint'] for c in comments]
        if "Từ khóa rủi ro" in sentiments:
            return "Bad Feature"
        
        good_count = sentiments.count("Tích cực")
        bad_count = sentiments.count("Tiêu cực")
        
        if good_count > bad_count:
            return "Good Feature"
        elif bad_count > good_count:
            return "Bad Feature"
        return "Neutral"
