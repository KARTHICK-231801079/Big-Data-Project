# This is the correct code for your LOCAL app.py file

import pandas as pd
import plotly
import plotly.express as px
import json
from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import NearestNeighbors
import numpy as np

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load Data and Build Models ---

print("Loading data...")
# Load dashboard data from the local file
silver_df = pd.read_csv("big_data_project.silver.sliver_data", quotechar='"') # Added the fix here too
# Load recommendation data from the local file
gold_df = pd.read_parquet("big_data_project.gold.gold_data")
print("Data loaded.")

# --- 3. Create Dashboard Charts ---

print("Creating dashboard charts...")
# Chart 1: Average Price by Category
avg_price_chart = px.bar(
    silver_df.groupby("category_name")["price"].mean().reset_index(),
    x="category_name",
    y="price",
    title="Average Price by Category"
)
chart1_json = json.dumps(avg_price_chart, cls=plotly.utils.PlotlyJSONEncoder)

# Chart 2: Items Sold by Category
bought_chart = px.pie(
    silver_df.groupby("category_name")["boughtInLastMonth"].sum().reset_index(),
    names="category_name",
    values="boughtInLastMonth",
    title="Items Sold by Category"
)
chart2_json = json.dumps(bought_chart, cls=plotly.utils.PlotlyJSONEncoder)
print("Charts created.")

# --- 4. Build Recommendation Model (Scikit-learn) ---

print("Building recommendation model...")
features_matrix = np.array(gold_df["features"].to_list())
model_nn = NearestNeighbors(metric='cosine', algorithm='brute')
model_nn.fit(features_matrix)
print("Model built successfully.")

# --- 5. Define Web Page Routes (Endpoints) ---

@app.route('/')
def dashboard():
    """Renders the main dashboard page."""
    # You MUST create the 'templates' folder and put 'index.html' inside it
    return render_template('index.html', chart1JSON=chart1_json, chart2JSON=chart2_json)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles recommendation requests from the search bar."""
    query = request.json['query'].lower()
    match = gold_df[gold_df['title'].str.lower().str.contains(query)]
    
    if match.empty:
        return jsonify({"error": "No product found with that name."}), 404
        
    target_index = match.index[0]
    target_vector = features_matrix[target_index].reshape(1, -1)
    
    distances, indices = model_nn.kneighbors(target_vector, n_neighbors=6)
    
    recommendations = []
    for i in indices.flatten()[1:]:
        rec = gold_df.iloc[i]
        recommendations.append({
            "title": rec["title"],
            "price": rec["price"],
            "stars": rec["stars"],
            "imgUrl": rec["imgUrl"],
            "productURL": rec["productURL"]
        })
        
    return jsonify(recommendations)

# --- 6. Run the App ---
if __name__ == '__main__':
    print("Starting Flask app... Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)