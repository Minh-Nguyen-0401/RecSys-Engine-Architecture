import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path

# Add the parent directory to sys.path to access existing modules
HM_TWO_STEP_RECO_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(HM_TWO_STEP_RECO_DIR))

from online_pipeline import run_online_recommend

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Paths to data files
INFERENCE_RESULTS_PATH = HM_TWO_STEP_RECO_DIR / 'two_tower_cg' / 'refactor' / 'output' / 'inference' / 'inference_results.parquet'
ARTICLES_PATH = HM_TWO_STEP_RECO_DIR / 'data' / 'articles.csv'
OUTPUT_DIR = HM_TWO_STEP_RECO_DIR / 'output'

# Load articles data for descriptions and image paths
def load_articles_data():
    articles_df = pd.read_csv(ARTICLES_PATH)
    articles_df['article_id'] = articles_df['article_id'].astype(str)
    return articles_df

# Get image path based on article_id
def get_image_path(article_id):
    article_id_str = str(article_id).zfill(10)  # Ensure 10 digits
    subfolder = article_id_str[:3]
    image_name = f"{article_id_str}.jpg"
    image_path = f"/images/{subfolder}/{image_name}"
    return image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_filter_options')
def get_filter_options():
    try:
        articles_df = load_articles_data()
        filters = {
            'product_type_name': sorted(articles_df['product_type_name'].dropna().unique().tolist()),
            'product_group_name': sorted(articles_df['product_group_name'].dropna().unique().tolist()),
            'department_name': sorted(articles_df['department_name'].dropna().unique().tolist()),
            'colour_group_name': sorted(articles_df['colour_group_name'].dropna().unique().tolist()),
            'graphical_appearance': sorted(articles_df['graphical_appearance_name'].dropna().unique().tolist())
        }
        return jsonify(filters)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_customers')
def get_customers():
    try:
        inference_df = pd.read_parquet(INFERENCE_RESULTS_PATH)
        customer_ids = inference_df['customer_id'].unique().tolist()
        return jsonify(customer_ids)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_recommendations/<customer_id>')
def get_recommendations(customer_id):
    try:
        reranked_path = OUTPUT_DIR / f'reranked_recommendations_{customer_id}.parquet'
        if not reranked_path.exists():
            return jsonify({'error': 'Recommendations not found for this customer'}), 404

        reranked_df = pd.read_parquet(reranked_path)
        if reranked_df.empty or 'predicted_article_ids' not in reranked_df.columns:
            return jsonify({'error': 'No recommendations available'}), 404

        article_ids = reranked_df['predicted_article_ids'].iloc[0].split(' ')[:300]
        articles_df = load_articles_data()

        # Apply filters
        filters = request.args
        filtered_articles = articles_df[articles_df['article_id'].isin(article_ids)]
        for key, value in filters.items():
            if value and key in filtered_articles.columns:
                filtered_articles = filtered_articles[filtered_articles[key] == value]

        recommendations = []
        for _, row in filtered_articles.iterrows():
            recommendations.append({
                'article_id': row['article_id'],
                'prod_name': row['prod_name'],
                'detail_desc': row['detail_desc'],
                'image_path': get_image_path(row['article_id'])
            })

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:path>')
def serve_image(path):
    # Serve images from the 'images' folder inside the webapp directory
    image_dir = os.path.join(app.root_path, 'images')
    print(f"Attempting to serve image: {path} from directory: {image_dir}")
    return send_from_directory(image_dir, path)

@app.route('/search', methods=['POST'])
def search():
    try:
        customer_id = request.form.get('customer_id')
        search_type = request.form.get('search_type')
        query = request.form.get('query', '')

        if search_type == 'image':
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            query = filepath

        run_online_recommend(customer_id, method=search_type, query=query)
        result_path = OUTPUT_DIR / f'final_rec_with_{search_type}.json'

        if not result_path.exists():
            return jsonify({'error': 'Search failed to generate results'}), 500

        with open(result_path, 'r') as f:
            results = json.load(f)

        result_df = pd.DataFrame(results)
        articles_df = load_articles_data()

        # Merge to get all details
        full_results = pd.merge(result_df, articles_df, on='article_id', how='left')

        # Handle column name conflict from merge. 'detail_desc' from articles_df (now _y) is preferred.
        if 'detail_desc_y' in full_results.columns:
            if 'detail_desc_x' in full_results.columns:
                full_results = full_results.drop(columns=['detail_desc_x'])
            full_results = full_results.rename(columns={'detail_desc_y': 'detail_desc'})

        # Apply filters
        filters = request.form
        for key, value in filters.items():
            if value and key in full_results.columns:
                full_results = full_results[full_results[key] == value]

        recommendations = []
        for _, row in full_results.head(100).iterrows(): # Limit to top 100
            recommendations.append({
                'article_id': row['article_id'],
                'prod_name': row['prod_name'],
                'detail_desc': row['detail_desc'],
                'image_path': get_image_path(row['article_id'])
            })

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
