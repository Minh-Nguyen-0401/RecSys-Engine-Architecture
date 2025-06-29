import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def search_by_text(query: str, reranked_df, article_text_df, top_k=10):
    """
    Step 4: Search for articles whose textual descriptions are similar to a given text query.
    """
    print("\n--- Running Step 4: Text Query Search ---")

    # 1. Get predicted article list
    user_articles = reranked_df['predicted_article_ids'].iloc[0].split(' ')
    candidates_text = article_text_df[article_text_df['article_id'].isin(user_articles)].copy()

    # 2. Apply TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(candidates_text['detail_desc'].fillna(""))

    # 3. Transform query to same TF-IDF space
    query_vec = vectorizer.transform([query])

    # 4. Compute cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    candidates_text['similarity'] = similarities

    # 5. Sort and return top results
    top_results = candidates_text.sort_values(by='similarity', ascending=False).head(top_k)
    print("Text query search complete.")
    return top_results[['article_id', 'detail_desc', 'similarity']]
