from typing import List

class Config:
    def __init__(self,
                 embedding_dimension: int,
                 batch_size: int,
                 learning_rate: float,
                 nb_epochs: int,
                 model_save_dir: str,
                 recall_k_values: List[int]):
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.model_save_dir = model_save_dir
        self.recall_k_values = recall_k_values

    def to_json(self):
        return {
            'embedding_dimension': self.embedding_dimension,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'nb_epochs': self.nb_epochs,
            'model_save_dir': self.model_save_dir,
            'recall_k_values': self.recall_k_values
        }


class Variables:
    ARTICLE_CATEG_VARIABLES: List[str] = ['article_id', 'product_type_name', 'product_group_name', 'colour_group_name',
                                          'department_name', 'index_name', 'section_name', 'garment_group_name']
    CUSTOMER_CATEG_VARIABLES: List[str] = ['customer_id', 'FN', 'Active', 'club_member_status',
                                           'fashion_news_frequency', 'age_interval', 'postal_code']
    IMG_EMB_VARIABLES: List[str] = [f"img_embd_{i}" for i in range(128)]
    ROLLING_FEATURES = [
        'customer_daily_spend', 'customer_daily_articles', 'customer_spend_30d', 'customer_articles_30d',
        'days_since_last_customer_trans'
    ]
    ALL_CATEG_VARIABLES = ARTICLE_CATEG_VARIABLES + CUSTOMER_CATEG_VARIABLES
