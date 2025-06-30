import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PRJ_ROOT = os.path.dirname(CUR_DIR)
sys.path.insert(0, PRJ_ROOT)

def main():
    
    articles_df = os.path.join(PRJ_ROOT, "data", "articles.csv")
    if not os.path.exists(articles_df):
        raise FileNotFoundError(f"Articles data not found at {articles_df}. Please check the path.")
    articles_df = pd.read_csv(articles_df)[["article_id", "detail_desc"]]
    articles_df["detail_desc"] = articles_df["detail_desc"].fillna("")
    articles_df["article_id"] = articles_df["article_id"].astype(str)
    print("Starting TF-IDF training...")

    # remove stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stopwords = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=5000, lowercase=True)
    tf_idf_matrix = vectorizer.fit_transform(articles_df['detail_desc'])

    # export the TF-IDF model
    tf_idf_model_path = os.path.join(CUR_DIR, "models", "tf_idf_model.pkl")
    with open(tf_idf_model_path, 'wb') as f:
        import pickle
        pickle.dump(vectorizer, f)
    print(f"TF-IDF model saved to {tf_idf_model_path}")

    # save matrix and corresponding article IDs but still in sparse format
    tf_idf_matrix_path = os.path.join(CUR_DIR, "output", "tf_idf_matrix.npz")
    from scipy import sparse
    sparse.save_npz(tf_idf_matrix_path, tf_idf_matrix)
    print(f"TF-IDF matrix saved to {tf_idf_matrix_path}")

    # save article IDs
    article_ids_path = os.path.join(CUR_DIR, "output", "article_ids.npy")
    article_id = articles_df["article_id"].values
    np.save(article_ids_path, article_id)
    print(f"Article IDs saved to {article_ids_path}")

if __name__ == "__main__":
    main()
    print("TF-IDF training completed successfully.")

