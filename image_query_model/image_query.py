import os
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms as v2
from transformers import AutoImageProcessor, SwinModel, SwinConfig
from huggingface_hub import PyTorchModelHubMixin
import torch.nn as nn
import torch.nn.functional as F

def search_by_image(image_path, reranked_df, image_embeddings_df, threshold = 0.3):
    print("\n--- Running Step 3: Image Search ---")

    print(f"Extracting embedding for image: {image_path}")
    device = torch.device('cpu')
    ckpt = "yainage90/fashion-image-feature-extractor"
    encoder_config = SwinConfig.from_pretrained(ckpt)
    image_processor = AutoImageProcessor.from_pretrained(ckpt)

    class ImageEncoder(nn.Module, PyTorchModelHubMixin):
        def __init__(self, encoder_config):
            super(ImageEncoder, self).__init__()
            self.swin = SwinModel(config=encoder_config)
            self.embedding_layer = nn.Linear(encoder_config.hidden_size, 128)

        def forward(self, image_tensor):
            features = self.swin(image_tensor).pooler_output
            embeddings = self.embedding_layer(features)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings

    encoder = ImageEncoder.from_pretrained(ckpt, encoder_config=encoder_config).to(device)
    encoder.eval()  # Set to evaluation mode

    transform = v2.Compose([
        v2.Resize((encoder_config.image_size, encoder_config.image_size)),
        v2.ToTensor(),
        v2.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().numpy()

    user_articles = reranked_df['predicted_article_ids'].iloc[0].split(' ')
    feed_embeddings = image_embeddings_df[image_embeddings_df['article_id'].isin(user_articles)]

    print("Calculating cosine similarity...")
    article_embeddings = feed_embeddings.drop('article_id', axis=1).values
    similarities = cosine_similarity(image_embedding, article_embeddings).flatten()

    feed_embeddings['similarity'] = similarities

    final_recommendations = feed_embeddings[feed_embeddings['similarity'] >= threshold]
    final_recommendations = final_recommendations[['article_id', 'similarity']]
    final_recommendations = final_recommendations.sort_values(by='similarity', ascending=False).reset_index(drop=True)
    print("Image search complete.")
    return final_recommendations