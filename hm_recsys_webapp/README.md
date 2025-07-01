# H&M Recommendation System Web App

This is a simple Flask-based web application to demonstrate the H&M recommendation system. It allows users to select a customer ID, view default product feeds, and search for products using text or image queries.

## Setup

1. Ensure you have Python installed on your system.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the recommendation data files are available in the specified paths as per the project structure.

## Running the Application

1. Navigate to the `hm_recsys_webapp` directory.
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open a web browser and go to `http://127.0.0.1:5000/` to access the application.

## Features

- **Customer Selection**: Choose a customer ID from a dropdown to view their personalized recommendations.
- **Product Feed**: Displays the top recommendations with product images and descriptions.
- **Search**: Perform text or image-based searches to find specific products.

## Project Structure

- `app.py`: Main Flask application with API endpoints.
- `templates/index.html`: HTML template for the frontend.
- `static/css/style.css`: Custom styles for the UI.
- `static/js/main.js`: JavaScript for interactive functionality.
- `static/uploads/`: Directory for temporarily storing uploaded images.
