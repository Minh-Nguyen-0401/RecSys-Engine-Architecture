import pandas as pd

def apply_candidate_filters(candidates_df, article_df, customer_df, price_range_ratio=0.2):
    """
    Filter out invalid candidates based on business rules.
    """
    print("Applying filtering rules...")

    # Merge for context
    df = candidates_df.copy()
    df = df.explode('predicted_article_ids')
    df = df.rename(columns={'predicted_article_ids': 'article_id'})

    df = df.merge(article_df, on='article_id', how='left')
    df = df.merge(customer_df[['customer_id', 'price_range']], on='customer_id', how='left')

    # Filter rules
    filters = (
        (df['is_in_stock'] == 1) & 
        (df['is_seasonal'] == 1) & 
        (df['location_available'] == 1) &
        (df['price'] >= df['price_range'] * (1 - price_range_ratio)) &
        (df['price'] <= df['price_range'] * (1 + price_range_ratio))
    )
    df = df[filters]

    # Re-aggregate article_ids
    filtered_df = df.groupby('customer_id')['article_id'].apply(lambda ids: ' '.join(ids)).reset_index()
    return filtered_df
