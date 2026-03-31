import os
import json
import uuid
import numpy as np
import umap.umap_ as umap
import hdbscan
import polars as pl
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

class SemanticEngine:
    """
    Semantic Engine using ChromaDB for storage, UMAP for dimensionality reduction,
    HDBSCAN for clustering, and Google Gemini for sentiment.
    Classifies clusters into "Good Feature", "Bad Feature", or "Neutral".
    """
    def __init__(self, api_key: str = None, sentiment_model: str = 'gemini-2.5-flash', embedding_model: str = 'gemini-embedding-001'):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("[Warning] GEMINI_API_KEY not provided. Semantic analysis may fail.")
        
        # Set API key for GenAI client if provided
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client()
            
        self.model_name = sentiment_model
        self.embedding_model = embedding_model
        
        # Initialize Vector Database (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path='./chroma_db')
        self.collection = self.chroma_client.get_or_create_collection(name='comments_gemini_3072')
        
    def _analyze_cluster_sentiment(self, comments: list[str]) -> dict:
        """
        Calls Gemini API to classify a sample of comments from a cluster.
        Returns a dictionary with counts for each category.
        """
        if not comments:
            return {"Good Feature": 0, "Bad Feature": 0, "Neutral": 0}
            
        prompt = f"""
        You are an expert Data Analyst and Sentiment Classifier.
        Analyze the following user comments and classify EACH comment into exactly one of these three categories:
        1. "Good Feature" (positive feedback, praise, user likes it)
        2. "Bad Feature" (negative feedback, complaints, bug reports, user dislikes it)
        3. "Neutral" (questions, irrelevant, or mixed without strong lean)
        
        Return ONLY a JSON object containing the total count for each category.
        Strictly use this format:
        {{
            "Good Feature": <count>,
            "Bad Feature": <count>,
            "Neutral": <count>
        }}
        
        Comments to analyze:
        {json.dumps(comments, ensure_ascii=False)}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            
            result_json = response.text
            counts = json.loads(result_json)
            
            return {
                "Good Feature": counts.get("Good Feature", 0),
                "Bad Feature": counts.get("Bad Feature", 0),
                "Neutral": counts.get("Neutral", 0)
            }
        except Exception as e:
            print(f"[SemanticEngine] Error calling Gemini API: {e}")
            return {"Good Feature": 0, "Bad Feature": 0, "Neutral": len(comments)}

    def analyze_comments(self, comments: list[str]) -> str:
        """
        Analyzes a list of comments by vectorizing them, reducing dimensions with UMAP,
        clustering them via HDBSCAN, and classifying each cluster via Gemini.
        """
        print("\n--- [Semantic Engine] Analyzing Social Sentiment with ChromaDB, UMAP, HDBSCAN & Gemini ---")
        
        if not comments:
            print("No comments found.")
            return "Neutral"
            
        print(f"Storing {len(comments)} comments in ChromaDB and generating embeddings...")
        
        # Parse text from dict format if needed
        parsed_comments = []
        for c in comments:
            if isinstance(c, dict) and 'text' in c:
                parsed_comments.append(c['text'])
            elif isinstance(c, str):
                parsed_comments.append(c)
            else:
                parsed_comments.append(str(c))
                
        # Give fresh IDs to new comments
        ids = [str(uuid.uuid4()) for _ in parsed_comments]
        
        # Get embeddings from configured model
        print(f"Generating embeddings using {self.embedding_model}...")
        gemini_embeddings = []
        try:
            # Batch API call or single depending on library version. Usually it supports list.
            # Processing in smaller batches to avoid payload limits
            batch_size = 100
            for i in range(0, len(parsed_comments), batch_size):
                batch = parsed_comments[i:i + batch_size]
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch
                )
                if isinstance(response.embeddings, list):
                    for emb in response.embeddings:
                        gemini_embeddings.append(emb.values)
                else:
                    gemini_embeddings.append(response.embeddings.values)
        except Exception as e:
            print(f"[SemanticEngine] Error generating embeddings: {e}")
            gemini_embeddings = [[0.0] * 3072 for _ in parsed_comments]
            
        # Add to Chroma with generated embeddings
        self.collection.add(
            documents=parsed_comments,
            embeddings=gemini_embeddings,
            ids=ids,
            metadatas=[{"source": "user_comment"} for _ in parsed_comments]
        )
        
        # Retrieve embeddings back for dimensionality reduction and clustering
        result = self.collection.get(ids=ids, include=['embeddings', 'documents'])
        embeddings = np.array(result['embeddings'])
        docs = result['documents']
        
        if len(embeddings) < 5:
             print("Not enough comments for UMAP/HDBSCAN clustering, skipping...")
             clusters = {0: docs}
        else:
            print(f"Applying UMAP to reduce dimensions from {embeddings.shape[1]} to 5...")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                reducer = umap.UMAP(n_neighbors=min(15, len(embeddings)-1), n_components=5, metric='cosine', random_state=42)
                reduced_embeddings = reducer.fit_transform(embeddings)
            
            print("Clustering reduced embeddings using HDBSCAN...")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min(5, len(reduced_embeddings)), metric='euclidean')
            labels = clusterer.fit_predict(reduced_embeddings)
            
            clusters = {}
            for label, doc in zip(labels, docs):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(doc)
                
        print(f"Found {len(clusters)} unique clusters/topics.")
        
        total_good = 0
        total_bad = 0
        total_neutral = 0
        
        for cluster_id, cluster_docs in clusters.items():
            # sample up to 10 comments per cluster for LLM (as per PRD)
            sample_docs = cluster_docs[:10]
            label_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise (Outliers)"
            print(f"{label_name} (size: {len(cluster_docs)}) -> Sending {len(sample_docs)} samples to Gemini...")
            
            # Analyze cluster sentiment via Gemini
            batch_result = self._analyze_cluster_sentiment(sample_docs)
            
            # Determine dominant sentiment of this cluster
            dominant = "Neutral"
            max_c = -1
            for k, v in batch_result.items():
                if v > max_c:
                    max_c = v
                    dominant = k
            
            print(f"  -> Cluster Dominant Sentiment: {dominant}")
            
            # Weight the sentiment by the full cluster size
            if dominant == "Good Feature":
                total_good += len(cluster_docs)
            elif dominant == "Bad Feature":
                total_docs = len(cluster_docs)
                # Ensure the full cluster size is added to total_bad
                total_bad += total_docs
            else:
                total_neutral += len(cluster_docs)
                
        print(f"Weighted Total by Cluster Size -> Good: {total_good}, Bad: {total_bad}, Neutral: {total_neutral}")
        
        if total_good > total_bad and total_good > total_neutral:
            reason = "Good Feature"
        elif total_bad > total_good and total_bad > total_neutral:
            reason = "Bad Feature"
        else:
            reason = "Neutral"
            
        print(f"Final Semantic Signal: {reason}")
        return reason
