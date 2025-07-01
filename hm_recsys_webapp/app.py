import os
import sys
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, send_from_directory
from pathlib import Path
import pandas as pd
import json
import subprocess
import os
import uuid

# Add the parent directory to sys.path to access existing modules
HM_TWO_STEP_RECO_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(HM_TWO_STEP_RECO_DIR))

from online_pipeline import run_online_recommend

app = Flask(__name__)

# Create a temporary folder for storing large search results
TEMP_FOLDER = Path(__file__).resolve().parent / 'temp_results'
TEMP_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.secret_key = 'a_super_secret_key_for_sessions'  # Required for session management

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Paths to data files
INFERENCE_RESULTS_PATH = HM_TWO_STEP_RECO_DIR / 'two_tower_cg' / 'refactor' / 'output' / 'inference' / 'inference_results.parquet'
ARTICLES_PATH = HM_TWO_STEP_RECO_DIR / 'data' / 'articles.csv'
OUTPUT_DIR = HM_TWO_STEP_RECO_DIR / 'output'

# Map frontend filter keys to backend dataframe column names
FILTER_COLUMN_MAP = {
    'product_type': 'product_type_name',
    'product_group': 'product_group_name',
    'department': 'department_name',
    'colour_group': 'colour_group_name',
    'graphical_appearance': 'graphical_appearance_name'
}

# Load articles data for descriptions and image paths
def get_dynamic_filter_options(df):
    options = {}
    for key, column in FILTER_COLUMN_MAP.items():
        if column in df.columns:
            options[key] = sorted(df[column].dropna().unique().tolist())
    return options

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
            'product_type': sorted(articles_df['product_type_name'].dropna().unique().tolist()),
            'product_group': sorted(articles_df['product_group_name'].dropna().unique().tolist()),
            'department': sorted(articles_df['department_name'].dropna().unique().tolist()),
            'colour_group': sorted(articles_df['colour_group_name'].dropna().unique().tolist()),
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

        article_ids = reranked_df['predicted_article_ids'].iloc[0].split(' ')[:5000]
        articles_df = load_articles_data()

        # Get the full set of recommended articles for this user to generate dynamic filters
        recommended_articles_df = articles_df[articles_df['article_id'].isin(article_ids)]
        recommended_articles_df = recommended_articles_df.set_index('article_id').reindex(article_ids).reset_index()

        dynamic_options = get_dynamic_filter_options(recommended_articles_df)

        # Now, apply filters from the request to the recommended articles
        filtered_articles = recommended_articles_df.copy()
        filters = request.args
        for key, value in filters.items():
            if value and key in FILTER_COLUMN_MAP:
                column_name = FILTER_COLUMN_MAP[key]
                if column_name in filtered_articles.columns:
                    filtered_articles = filtered_articles[filtered_articles[column_name] == value]

        recommendations = []
        for _, row in filtered_articles.iterrows():
            recommendations.append({
                'article_id': row['article_id'],
                'prod_name': 'N/A' if pd.isna(row['prod_name']) else row['prod_name'],
                'detail_desc': '' if pd.isna(row['detail_desc']) else row['detail_desc'],
                'image_path': get_image_path(row['article_id'])
            })

        return jsonify({'recommendations': recommendations, 'filter_options': dynamic_options})
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
    # Clean up previous temp file if it exists for this session
    if 'last_results_path' in session:
        results_path = session.pop('last_results_path', None)
        if results_path and os.path.exists(results_path):
            try:
                os.remove(results_path)
                app.logger.info(f"Removed old temp file: {results_path}")
            except OSError as e:
                app.logger.error(f"Error removing old temp file: {e}")

    session.pop('last_search_type', None)
    try:
        customer_id = request.form.get('customer_id')
        search_type = request.form.get('search_type')
        
        if search_type == 'text':
            query = request.form.get('query', 'shirt').strip()
            app.logger.info(f"Received text search query: '{query}'")

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

        # Generate dynamic options from the full, unfiltered results
        dynamic_options = get_dynamic_filter_options(full_results)

        # Save results to a temporary file and store the path in the session
        results_filename = f"{uuid.uuid4()}.parquet"
        results_filepath = str(TEMP_FOLDER / results_filename)
        full_results.to_parquet(results_filepath)
        session['last_results_path'] = results_filepath
        app.logger.info(f"Saved search results to {results_filepath}")
        app.logger.info(f"Set session keys after search: {list(session.keys())}")

        # Apply filters from the initial search request
        filtered_results = full_results.copy()
        filters = request.form
        for key, value in filters.items():
            if value and key in FILTER_COLUMN_MAP:
                column_name = FILTER_COLUMN_MAP[key]
                if column_name in filtered_results.columns:
                    filtered_results = filtered_results[filtered_results[column_name] == value]

        recommendations = []
        for _, row in filtered_results.head(2000).iterrows():
            recommendations.append({
                'article_id': row['article_id'],
                'prod_name': 'N/A' if pd.isna(row['prod_name']) else row['prod_name'],
                'detail_desc': '' if pd.isna(row['detail_desc']) else row['detail_desc'],
                'image_path': get_image_path(row['article_id'])
            })

        return jsonify({'recommendations': recommendations, 'filter_options': dynamic_options})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/filter_results', methods=['POST'])
def filter_results():
    try:
        app.logger.info(f"Filter request. Session keys: {list(session.keys())}")
        if 'last_results_path' not in session:
            return jsonify({'error': 'No search results to filter. Please perform a new search.'}), 400

        results_path = session['last_results_path']
        if not os.path.exists(results_path):
            return jsonify({'error': 'Search results have expired or been cleared. Please perform a new search.'}), 400

        results_df = pd.read_parquet(results_path)

        # Re-generate dynamic options from the full cached results to keep the session small
        dynamic_options = get_dynamic_filter_options(results_df)

        # Apply new filters from the request
        filtered_results = results_df.copy()
        filters = request.form
        for key, value in filters.items():
            if value and key in FILTER_COLUMN_MAP:
                column_name = FILTER_COLUMN_MAP[key]
                if column_name in filtered_results.columns:
                    filtered_results = filtered_results[filtered_results[column_name] == value]

        recommendations = []
        for _, row in filtered_results.head(2000).iterrows():
            recommendations.append({
                'article_id': row['article_id'],
                'prod_name': 'N/A' if pd.isna(row['prod_name']) else row['prod_name'],
                'detail_desc': '' if pd.isna(row['detail_desc']) else row['detail_desc'],
                'image_path': get_image_path(row['article_id'])
            })

        return jsonify({'recommendations': recommendations, 'filter_options': dynamic_options})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_search', methods=['POST'])
def clear_search():
    try:
        # Clear the last search results path and type from the session
        if 'last_results_path' in session:
            results_path = session.pop('last_results_path', None)
            if results_path and os.path.exists(results_path):
                try:
                    os.remove(results_path)
                    app.logger.info(f"Removed temp file: {results_path}")
                except OSError as e:
                    app.logger.error(f"Error removing temp file: {e}")

        session.pop('last_search_type', None)
        app.logger.info(f"Cleared search. Session keys: {list(session.keys())}")
        return jsonify({'status': 'success'})
    except Exception as e:
        app.logger.error(f"Error clearing search: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
