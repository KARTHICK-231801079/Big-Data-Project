import pandas as pd
import json
from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix 

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load Data and Build Model (This runs only once at startup) ---

print("Loading data...")
# Load recommendation data (Parquet file)
# We no longer need silver_data.csv for this page
gold_df = pd.read_parquet("gold_data.parquet")
print("Data loaded.")

# --- 3. Build Recommendation Model (Scikit-learn) ---
# (This section is renamed from #4)

print("Building recommendation model...")

def convert_spark_vectors_to_scipy(spark_vectors):
    """
    Converts a list of Spark SparseVector dictionaries 
    into a Scipy Compressed Sparse Row (CSR) matrix.
    """
    num_rows = len(spark_vectors)
    if num_rows == 0:
        return csr_matrix((0, 0))

    # Find the first non-null vector to get the size
    first_vec = next((v for v in spark_vectors if v is not None), None)
    if first_vec is None:
         return csr_matrix((0, 0))
    
    vector_size = first_vec['size']
    
    data = []
    indices = []
    indptr = [0] # This is for csr_matrix format

    for vec in spark_vectors:
        if vec is None:
            indptr.append(len(data))
            continue
            
        data.extend(vec['values'])
        indices.extend(vec['indices'])
        indptr.append(len(data))
        
    return csr_matrix((data, indices, indptr), shape=(num_rows, vector_size))


# 1. Get the list of vector dictionaries from the DataFrame
spark_vectors_list = gold_df["features"].to_list()

# 2. Convert this list into a Scipy CSR (Compressed Sparse Row) matrix
features_matrix = convert_spark_vectors_to_scipy(spark_vectors_list)

# 3. Build the model
model_nn = NearestNeighbors(metric='cosine', algorithm='brute')
model_nn.fit(features_matrix)
print("Model built successfully.")


# --- 4. Define Web Page Routes (Endpoints) ---

@app.route('/')
def home():
    """Renders the main search page."""
    # We don't need to pass any charts, just render the HTML
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles recommendation requests from the search bar."""
    
    query = request.json['query'].lower()
    
    # Find all products that match the query
    matches = gold_df[gold_df['title'].str.lower().str.contains(query, na=False)]
    
    if matches.empty:
        return jsonify({"error": "No product found with that name."}), 404
        
    # Get the vector of the *first* matching product
    target_index = matches.index[0]
    target_vector = features_matrix[target_index]
    
    # Find the 6 nearest neighbors
    distances, indices = model_nn.kneighbors(target_vector, n_neighbors=6)
    
    recommendations = []
    # Skip the first one (indices.flatten()[0]) because it's the product itself
    for i in indices.flatten()[1:]:
        rec = gold_df.iloc[i]
        recommendations.append({
            # Use .get() to avoid errors if a column is missing
            "title": rec.get("title", "No Title"),
            "price": rec.get("price", 0.0),
            "stars": rec.get("stars", 0),
            "imgUrl": rec.get("imgUrl", ""),
            "productURL": rec.get("productURL", "#")
        })
        
    return jsonify(recommendations)

# --- 5. Run the App ---
if __name__ == '__main__':
    print("Starting Flask app... Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)