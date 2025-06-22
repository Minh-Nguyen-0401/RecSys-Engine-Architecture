from PIL import Image
from pathlib import Path
import os 
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as v2
from transformers import AutoImageProcessor, SwinModel, SwinConfig
from huggingface_hub import PyTorchModelHubMixin
import logging
import warnings
warnings.filterwarnings("ignore")


CUR_DIR = Path(__file__).resolve().parent

IMAGE_FOLDER = r"D:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\data\images"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S', 
                    handlers= [
                        logging.FileHandler(CUR_DIR / 'logs' / 'image_feature_extraction.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
def get_image_paths(image_folder_path = IMAGE_FOLDER):
    if not image_folder_path:
        return []
    return [str(f) for f in Path(image_folder_path).rglob("**/*.jpg")]

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")

    ckpt = "yainage90/fashion-image-feature-extractor"
    encoder_config = SwinConfig.from_pretrained(ckpt)
    encoder_image_processor = AutoImageProcessor.from_pretrained(ckpt)

    class ImageEncoder(nn.Module, PyTorchModelHubMixin):
        def __init__(self):
            super(ImageEncoder, self).__init__()
            self.swin = SwinModel(config=encoder_config)
            self.embedding_layer = nn.Linear(encoder_config.hidden_size, 128)

        def forward(self, image_tensor):
            features = self.swin(image_tensor).pooler_output
            embeddings = self.embedding_layer(features)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings

    encoder = ImageEncoder().from_pretrained('yainage90/fashion-image-feature-extractor').to(device)
    logger.info(f"Encoder loaded with config: {encoder_config}")

    transform = v2.Compose([
        v2.Resize((encoder_config.image_size, encoder_config.image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=encoder_image_processor.image_mean, std=encoder_image_processor.image_std),
    ])
    logger.info("Image transformation pipeline created.")

    image_embeddings = defaultdict(np.ndarray)
    for image_path in get_image_paths():
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        article_id = os.path.basename(image_path).split('.')[0]
        logger.info(f"Processing image: {image_path} with article_id: {article_id}")
        with torch.no_grad():
            embedding = encoder(image.unsqueeze(0).to(device)).cpu().numpy()
            image_embeddings[article_id] = embedding
        logger.info(f"Finished {len(image_embeddings)} images...")

    # Save embeddings to a file
    output_files = os.path.join(CUR_DIR, 'input', 'image_embeddings.npz')
    np.savez_compressed(output_files, **image_embeddings)
    logger.info(f"Image embeddings saved to {output_files}")

if __name__ == "__main__":
    main()