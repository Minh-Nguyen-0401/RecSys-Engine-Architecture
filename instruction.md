# 🛍️ H&M Two-Step Recommendation Pipeline

This document provides instructions for running the two-step recommendation pipeline and the accompanying Flask web application. The system first generates candidate articles using a Two-Tower model, re-ranks them using an MMOE model, and finally offers a web interface for personalized recommendations, including both **image-based** and **text-based** query search.

---

## 🚀 Web Application

A Flask-based web application provides an interactive interface to the recommendation system.

### ✨ Features

- **Customer Selection:** Select a customer ID to view their personalized feed.
- **Default Recommendations:** View a feed of reranked recommendations for the selected customer.
- **Text Search:** Search for products using a text query (e.g., "blue shirt").
- **Image Search:** Upload an image to find visually similar products.
- **Dynamic Filtering:** Filter search results by product type, color, department, etc.
- **Clear Search:** Easily clear search results and return to the default feed.

### ⚙️ How to Run the Web App

1. **Navigate to the webapp directory:**

   ```bash
   cd hm_recsys_webapp
   ```

2. **Activate your conda environment:**

   ```bash
   conda activate <your_env_name>
   ```

3. **Run the Flask application:**

   ```bash
   flask run
   ```

4. **Open the application in your browser:**

   [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📐 Pipeline Architecture

The pipeline is structured in **three key stages**:

1. **Candidate Generation (Two-Tower model)**
2. **Re-ranking (MMOE model)**
3. **Query Search**
   - By Image
   - By Text

```mermaid
graph TD
    subgraph "Offline Phase (Run Periodically)"
        A[Start] --> B(Two-Tower Candidate Generation);
        B --> C[Candidate Filtering];
        C --> D[inference_results.parquet];
    end

    subgraph "Online Phase (Real-time)"
        E(Load Candidates) --> D;
        E --> F(MMOE Ranker);
        F --> G(Initial Feed);
        G --> H[reranked_recommendations.parquet];
    end

    subgraph "Query Search (Optional)"
        I[User Image Upload] --> J[Extract Embedding];
        J --> K[Cosine Similarity w/ Article Embeddings];
        K --> L[Filtered Feed by Image];

        M[User Text Query] --> N[TF-IDF Similarity];
        N --> O[Filtered Feed by Text];
    end
```

---

## ⚙️ How to Run the Pipeline

### ✅ Prerequisites

Ensure the following are available:

- ✅ Conda environment with required dependencies
- ✅ Dataset files in `data/`:
  - `articles.csv`, `customers.csv`, `transactions_train.csv`
- ✅ Trained models:
  - Two-Tower: `output/models/model_<version>/model_weights.h5`
  - MMOE Ranker: `output/parquet/model`
- ✅ Precomputed image embeddings: `data/image_embeddings.parquet`

---

### 🔹 Step 1: Generate and Filter Candidates

Navigate to the Two-Tower folder and run inference. A candidate filter module has been added internally in `__inference__.py` for optional rule-based filtering.

```bash
cd two_tower_cg/refactor

# Run inference with optional threshold or filter logic
python __inference__.py -mv v2 --threshold 0.3 --top_k 5000
```

📦 Output: `output/inference/inference_results.parquet`
✨ *Note: This step now includes candidate filtering before saving results.*

---

### 🔹 Step 2: Re-rank Using MMOE

The re-ranking pipeline uses the inference results and customer/article features to produce a final feed per user.

```bash
# Run full pipeline from project root
python recommendation_pipeline.py
```

#### Optional: Re-rank for a specific customer

```bash
python recommendation_pipeline.py --customer_id <valid_customer_id>
```

📦 Output: `output/reranked_recommendations.parquet`

---

### 🔹 Step 3: Run Query Search (Image + Text)

#### 🖼️ Image-Based Search

If `test_img.jpg` and `image_embeddings.parquet` are present:

```bash
# Automatically triggered inside recommendation_pipeline.py
# Top similar articles by visual embedding
```

#### 🔤 Text-Based Search

You can modify the example query inside `recommendation_pipeline.py`, e.g.:

```python
example_query = "floral summer dress"
```

📦 Output:

- Console print of top articles with highest similarity
- `output/final_recs_text_query.json`

---

## 📂 Output Files

| File | Description |
|------|-------------|
| `inference/inference_results.parquet` | Top-k articles per customer from Two-Tower |
| `reranked_recommendations.parquet` | Final re-ranked articles by MMOE |
| `final_recs_img_query.json` | Final articles filtered by image similarity |
| `final_recs_text_query.json` | Final articles filtered by text query |

---

## ✅ Summary of New Features

| Feature | Location | Description |
|--------|----------|-------------|
| 🔎 `filter_candidates` | `__inference__.py` | Optional postprocessing step after Two-Tower output |
| 🔤 `search_by_text()` | `recommendation_pipeline.py` | TF-IDF cosine similarity search on `detail_desc` |
| 🧠 Improved modularity  | N/A                          | Image + Text query unified in a single flow      |
| 🌐 **Web Application**  | `hm_recsys_webapp/`          | Interactive Flask app for search and recommendations |

---

## 🧪 TODO / Future Work

- Add support for semantic search with pretrained Sentence-BERT
- Store embeddings and features in vector DBs (e.g., FAISS, Qdrant)
- Session-based recommendations (Click, Cart, Purchase)