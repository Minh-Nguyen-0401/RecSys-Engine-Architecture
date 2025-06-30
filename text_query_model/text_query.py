import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import numpy as np
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CUR_DIR, "tf_idf_model.pkl")

def search_by_text(query: str, reranked_df, article_df, threshold = 0.2):
    """
    Step 4: Search for articles whose textual descriptions are similar to a given text query.
    """
    print("\n--- Running Step 4: Text Query Search ---")

    # 1. Get predicted article list
    user_articles = reranked_df['predicted_article_ids'].iloc[0].split(' ')
    candidates_df = article_df[article_df['article_id'].isin(user_articles)]

    # 2. Load the pre-trained TF-IDF model and the transformed TF-IDF matrix
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"TF-IDF model not found at {MODEL_DIR}. Please run the training script first.")
    with open(MODEL_DIR, 'rb') as f:
        vectorizer = pickle.load(f)
    tfidf_matrix_path = os.path.join(CUR_DIR, "tf_idf_matrix.npz")
    if not os.path.exists(tfidf_matrix_path):
        raise FileNotFoundError(f"TF-IDF matrix not found at {tfidf_matrix_path}. Please run the training script first.")

    tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
    article_ids = np.load(os.path.join(CUR_DIR, "article_ids.npy"))

    # filter the matrix to only include candidates
    mask = np.isin(article_ids, candidates_df['article_id'].values)
    tfidf_matrix = tfidf_matrix[mask, :]
    article_ids = article_ids[mask]

    # 3. Transform query to same TF-IDF space
    query_vec = vectorizer.transform([query])

    # 4. Compute cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    final_candidates = pd.DataFrame({
        'article_id': article_ids,
        'similarity': similarities
    })
    final_candidates = final_candidates[final_candidates['similarity'] >= threshold].reset_index(drop=True)
    # final_candidates_text = final_candidates.merge(candidates_df[['article_id', 'detail_desc']], on='article_id', how='left')

    # keep the filtered candidates but with original ranking/order
    final_candidates["org_order"] = final_candidates["article_id"].apply(
        lambda x: user_articles.index(x) if x in user_articles else -1
    )
    final_candidates = final_candidates.sort_values(by='org_order').drop(columns=['org_order']).reset_index(drop=True)

    print("Text query search complete.")
    return final_candidates
