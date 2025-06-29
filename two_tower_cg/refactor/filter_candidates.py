import pandas as pd

def apply_candidate_filters(candidates_df, article_df, price_range_ratio=0.2):
    """
    Filter out invalid candidates based on business rules:
    - Article is in stock, seasonal, available
    - Article price within ±X% of user's latest transaction price
    """

    print("Applying candidate filtering rules...")

    # Step 0: Add dummy metadata 
    article_df = article_df.copy()
    for col in ['is_in_stock', 'is_seasonal', 'location_available']:
        if col not in article_df.columns:
            article_df[col] = 1  # giả sử luôn hợp lệ

    # Step 1: Explode candidates
    df = candidates_df.copy()
    df = df.explode('predicted_article_ids')
    df = df.rename(columns={'predicted_article_ids': 'article_id'})
    df['article_id'] = df['article_id'].astype(str)
    df['customer_id'] = df['customer_id'].astype(str)

    # Step 2: Merge article metadata
    article_df['article_id'] = article_df['article_id'].astype(str)
    df = df.merge(article_df[['article_id', 'is_in_stock', 'is_seasonal', 'location_available']],
                  on='article_id', how='left')

    # Step 3: Load transactions
    transactions = pd.read_csv(r"D:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\hm-two-step-reco\data\transactions_train.csv",
                        usecols=['article_id', 'customer_id', 't_dat', 'price'])
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["customer_id"] = transactions["customer_id"].astype(str)
    transactions["article_id"] = transactions["article_id"].astype(str)

    # Step 4: Get latest transaction per customer (for price anchoring)
    latest_tx = transactions.sort_values("t_dat").groupby("customer_id").tail(1)
    latest_tx = latest_tx[["customer_id", "price"]].rename(columns={"price": "last_price"})
    df = df.merge(latest_tx, on="customer_id", how="left")

    # Step 5: Compute avg price per article (as proxy for current article price)
    article_price_df = transactions.groupby("article_id")["price"].mean().reset_index()
    article_price_df = article_price_df.rename(columns={"price": "article_price"})
    df = df.merge(article_price_df, on="article_id", how="left")

    # Step 6: Apply filters
    filters = (
        (df['is_in_stock'] == 1) &
        (df['is_seasonal'] == 1) &
        (df['location_available'] == 1) &
        (df['article_price'] >= df['last_price'] * (1 - price_range_ratio)) &
        (df['article_price'] <= df['last_price'] * (1 + price_range_ratio))
    )
    df = df[filters]

    # Step 7: Regroup
    filtered_df = df.groupby('customer_id')['article_id'].apply(lambda ids: ' '.join(ids)).reset_index()
    return filtered_df