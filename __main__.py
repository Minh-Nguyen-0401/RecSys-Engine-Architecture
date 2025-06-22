import logging
from src.data import load_data
from src.preprocess import preprocess
from src.train import build_model, run_training, save_model
from src.custom_recall import CustomRecall
from src.custom_cross_entropy_loss import CustomCrossEntropyLoss
from src.config import Config, Variables
import warnings
import os
import tensorflow as tf
from pathlib import Path
warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
FINAL_OUTPUT_DIR = CURRENT_DIR / 'output' / 'models'
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# Logging configuration
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
    )


def run_all(config: Config):
    # Extract
    train_df, val_df, articles_df = load_data()

    # Preprocess
    preprocessed_hm_data = preprocess(train_df, val_df, articles_df, batch_size=config.batch_size)

    # Build model
    model = build_model(preprocessed_hm_data, config)

    # Train & Saved model
    train_history = run_training(model, preprocessed_hm_data, config)
    save_model(model, config)
    
    return train_history
    

if __name__ == '__main__':
    logger.info("Starting training pipeline...")
    MODEL_VER_DIR = os.path.join(FINAL_OUTPUT_DIR, 'model_v2')
    os.makedirs(MODEL_VER_DIR, exist_ok=True)   
    config = Config(embedding_dimension=128,
                    batch_size=512,
                    learning_rate=0.05,
                    nb_epochs=4,
                    model_save_dir=MODEL_VER_DIR,
                    recall_k_values=[100, 500, 1000, 5000])
    history = run_all(config)
    results = history.history

    import json
    with open(os.path.join(MODEL_VER_DIR, 'history.json'), 'w') as f:
        json.dump(results, f)
    
    logger.info("Training finished. Results saved to 'history.json'")