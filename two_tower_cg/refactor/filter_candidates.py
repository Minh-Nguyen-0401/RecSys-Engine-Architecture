import pandas as pd
import json
import pytz
def apply_candidate_filters(candidates_df, article_df, price_range_ratio=0.2):
    """
    Filter out invalid candidates based on business rules:
    - Article is in stock, seasonal, available
    - Article price within ±X% of user's latest transaction price
    """

    print("Applying candidate filtering rules...")

    # Load seasonal group of articles
    seasonal_articles = json.load(open(r"D:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\hm_recsys_core\two_tower_cg\refactor\output\parquet\product_seasonal_group.json", "r"))
    winter_autumn_articles = seasonal_articles.get("winter_autumn", [])
    summer_spring_articles = seasonal_articles.get("summer_spring", [])
    all_season_articles = seasonal_articles.get("all_seasons", [])

    # Add seasonal information to article_df
    VN_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    current_season = "summer_spring" if pd.Timestamp.now(tz=VN_tz).month in [3, 4, 5, 6, 7, 8] else "winter_autumn"
    article_df = article_df.copy()
    # article_df["seasonal_group"] = article_df["product_type_name"].apply(lambda x: "summer_spring" if x in summer_spring_articles else
    #                                                           ("winter_autumn" if x in winter_autumn_articles else "all_seasons" if x in all_season_articles else "unknown"))
    # article_df["is_seasonal"] = article_df["seasonal_group"].apply(lambda x: 1 if x == current_season or x == "all_seasons" else 0)
    article_df["is_seasonal"] = 1
    article_df["is_in_stock"] = 1  # Giả sử tất cả sản phẩm đều có sẵn
    article_df["location_available"] = 1  # Giả sử tất cả sản phẩm đều có sẵn tại vị trí của khách hàng


    # Step 1: Explode candidates
    df = candidates_df.copy()
    df["predicted_article_ids"] = (
        df["predicted_article_ids"]
        .str.strip()
        .str.split(r"\s+")
    )
    df = df.explode('predicted_article_ids')
    df = df.rename(columns={'predicted_article_ids': 'article_id'})
    df['article_id'] = df['article_id'].astype(str)
    df['customer_id'] = df['customer_id'].astype(str)

    # Step 2: Merge article metadata
    article_df['article_id'] = article_df['article_id'].astype(str)
    df = df.merge(article_df[['article_id', 'is_in_stock', 'is_seasonal', 'location_available']],
                  on='article_id', how='left')

    transactions = pd.read_parquet(r"D:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\hm_recsys_core\two_tower_cg\refactor\output\parquet\raw_transactions.parquet")
    # for each customer, get their maximum price within the last 10 transactions
    transactions['customer_id'] = transactions['customer_id'].astype(str)
    transactions['article_id'] = transactions['article_id'].astype(str)
    transactions = transactions.sort_values(by=['customer_id', 't_dat'], ascending=[True, False])
    latest_trans = transactions.groupby('customer_id').head(10)
    price_range = latest_trans.groupby('customer_id').agg(
        min_price=('price', 'min'),
        max_price=('price', 'max'),
    ).reset_index()
    df = df.merge(price_range, on="customer_id", how="left")

    # Step 5: Compute avg price per article (as proxy for current article price)
    article_price_df = transactions.groupby("article_id")["price"].mean().reset_index()
    article_price_df = article_price_df.rename(columns={"price": "article_price"})
    df = df.merge(article_price_df, on="article_id", how="left")

    # Step 6: Apply filters
    is_in_stock_filter = (df['is_in_stock'] == 1).fillna(False)
    is_seasonal_filter = (df['is_seasonal'] == 1).fillna(False)
    location_available_filter = (df['location_available'] == 1).fillna(False)

    price_min_filter = (df['article_price'] >= df['min_price'] * (1 - price_range_ratio)).fillna(True)
    price_max_filter = (df['article_price'] <= df['max_price'] * (1 + price_range_ratio)).fillna(True)

    filters = (
        is_in_stock_filter &
        is_seasonal_filter &
        location_available_filter &
        price_min_filter &
        price_max_filter
    )
    df = df[filters]
    # Step 7: Regroup
    filtered_df = df.groupby('customer_id')['article_id'].apply(lambda ids: ' '.join(ids)).reset_index()
    filtered_df = filtered_df.rename(columns={'article_id': 'predicted_article_ids'})
    return filtered_df