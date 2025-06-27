from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TextQueryRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_text(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.numpy()

    def search(self, query, article_descriptions_df, threshold=0.4):
        query_vec = self.embed_text(query)
        article_vecs = np.stack(article_descriptions_df['embedding'].values)
        sims = cosine_similarity(query_vec, article_vecs).flatten()
        article_descriptions_df['similarity'] = sims
        return article_descriptions_df[article_descriptions_df['similarity'] > threshold].sort_values(by='similarity', ascending=False)
