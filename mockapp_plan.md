# Mock App Plan for H&M Recommendation System Web App Plan


## 1. Project Overview

- **Objective:** Create a simple Flask web application to demonstrate the H&M recommendation system.

- **Key Features:**

### Backend (Flask)

- **API Endpoints:**

```json
{
    "/get_customers": "Returns a list of all unique customer_ids.",
    "/get_recommendations/<customer_id>": "Returns a paginated list of default recommendations for a given customer.",
    "/search": "Handles text and image search queries and returns ranked article_ids.",
    "/images/<subfolder>/<filename>": "Serves product images."
}
```

### Frontend (HTML, CSS, JavaScript)

- **UI Components:**

### Data Integration

- **Customer Data:**

### User Interaction Flow

- **Initial Load:**

- **Customer Selection:**

- **Pagination:**

- **Text Search:**

- **Image Search:**

- **Clear Search:**

## 2. File Structure

- **`hm_recsys_webapp/`**

## 3. Implementation Steps

- **Step 1: Setup Flask App**

- **Step 2: Backend Logic**

- **Step 3: Frontend Development**

## App Structure

### 1. Directory Structure

```markdown
hm_recsys_webapp/
│
├── app.py                  # Main Flask application
├── templates/
│   └── index.html          # Main HTML template for the frontend
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles
│   ├── js/
│   │   └── main.js         # JavaScript for interactivity
│   └── uploads/            # Temporary storage for uploaded images
├── requirements.txt         # Dependencies
└── README.md               # Basic documentation for setup and run
```

### 2. Key Components

#### Backend (app.py)
- **Flask API Endpoints**:
  - `/`: Serve the main page with a dropdown for customer selection and display default feeds.
  - `/get_customers`: Endpoint to fetch available customer IDs from `inference_results.parquet`.
  - `/get_recommendations/<customer_id>`: Fetch and display recommendations for the selected customer from `reranked_recommendations_{customer_id}.parquet`.
  - `/search`: Handle text or image search queries, invoking the appropriate function from `online_pipeline.py`.
- **Data Integration**:
  - Read customer IDs from `inference_results.parquet`.
  - Load product details (description and image paths) from `articles.csv`.
  - Temporarily store uploaded images for processing.

#### Frontend (index.html, style.css, main.js)
- **UI Elements**:
  - **Dropdown**: For selecting customer ID.
  - **Feed Display Area**: Grid layout to show product images, IDs, and descriptions.
  - **Search Bar**: Input field for text queries and a file upload button for images.
- **Interactivity**:
  - Use JavaScript to fetch customer IDs on page load.
  - Update feed display when a new customer is selected.
  - Handle search submissions, displaying results in the feed area.

### 3. Functionality

#### Customer Selection
- On page load, fetch customer IDs from `/get_customers` and populate the dropdown.
- Default to the first customer ID or a predefined one (e.g., 'aa51fd04db21c0d2620a351dc5b94b704922d674b1c52a37225dd25a7a166ee0').
- When a customer is selected, fetch recommendations via `/get_recommendations/<customer_id>`.

#### Default Feed Display
- Display products from `reranked_recommendations_{customer_id}.parquet`.
- For each `article_id` in `predicted_article_ids` (split by space), fetch `detail_desc` from `articles.csv`.
- Construct image path as `D:\Study\UNIVERSITY\THIRD YEAR\Business Analytics\final assignment\data\images\<subfolder>\0<article_id>.jpg`.
  - Determine `<subfolder>` by possibly using the first few digits of `article_id` or a mapping if provided.
- Show image, article ID, and description in a card layout.

#### Search Functionality
- **Text Search**: User inputs text (e.g., 'long trousers'), submits via `/search` with `method=text`.
- **Image Search**: User uploads an image, submits via `/search` with `method=image`.
- Both call the appropriate function in `online_pipeline.py`, passing the customer ID and query.
- Display results similarly to the default feed, using data from `final_rec_with_<method>.json`.

### 4. Integration with Existing Code
- **Offline Recommendations**: Use data from `reranked_recommendations_{customer_id}.parquet` for initial feeds.
- **Online Pipeline**: Invoke `run_online_recommend` from `online_pipeline.py` for search queries, ensuring it uses the correct customer ID and saves results to a JSON file.
- Ensure `articles.csv` is accessible for product details.

### 5. UI Mockup
- **Header**: Title 'H&M Recommendation System'.
- **Main Section**: 
  - Top: Dropdown for customer selection.
  - Middle: Search bar with text input and image upload option.
  - Bottom: Grid of product cards (image, ID, description).
- **Footer**: Simple text with project info.

### 6. Challenges and Considerations
- **Image Path Resolution**: Need to confirm how subfolders are determined for image paths. Assuming it might be based on `article_id` prefix.
- **Performance**: Loading many images or large datasets might be slow. Consider pagination or limiting displayed items.
- **Security**: Sanitize inputs for text search and secure file uploads for images.

### 7. Development Steps
1. Set up Flask project structure and install dependencies.
2. Create backend logic to read customer IDs and recommendations.
3. Develop frontend to display data dynamically.
4. Implement search functionality integrating with `online_pipeline.py`.
5. Test with sample data and refine UI/UX.

## Questions for Clarification
- How is the subfolder for image paths determined from `article_id`? Is it based on the first few characters or a separate mapping?
- Are there any specific limits or preferences for the number of products displayed in the feed?
- Should the search results overwrite the current feed or be displayed separately?
- Any specific UI design preferences or color schemes to match H&M branding?

This plan will guide the development of the web app. Once approved or after addressing any feedback, I will proceed with the implementation.
