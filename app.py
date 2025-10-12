from flask import Flask, render_template, request, jsonify
import pickle
import os

# Import for fuzzy matching
from thefuzz import process 

# --- Flask Setup ---
# Initialize Flask app
app = Flask(__name__)

# --- AI Model Loading ---
MODEL_FILE = 'model.pkl'

# Load the pre-trained model components
try:
    with open(MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)
        df = model_data['df']
        indices = model_data['indices']
        cosine_sim = model_data['cosine_sim']
    print("AI Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: {MODEL_FILE} not found. Run model_builder.py first!")
    df = None
    indices = None
    cosine_sim = None
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    df = None

# --- Recommendation Function ---
def get_recommendations(title, cosine_sim_matrix, df_data, indices_map, num_rec=10):
    """Generates a list of movie recommendations."""
    if df_data is None:
        return []

    # Check if the movie exists
    if title not in indices_map:
        return []

    # Get the index of the movie that matches the title
    idx = indices_map[title]

    # Get the pairwise similarity scores for the movie
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort the movies based on the similarity scores (most similar first)
    # [1:] skips the first result, which is the movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_rec + 1]

    # Get the movie indices and titles
    movie_indices = [i[0] for i in sim_scores]
    recommended_titles = df_data['title'].iloc[movie_indices].tolist()

    return recommended_titles

# --- Flask Routes ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    # Pass all unique movie titles to the frontend for a simple autocomplete/dropdown
    movie_titles = df['title'].tolist() if df is not None else []
    return render_template('index.html', movie_titles=movie_titles)

# --- Helper Function for Fuzzy Matching ---
def find_best_match(user_title, df_data):
    """
    Uses thefuzz to find the closest matching title in the DataFrame.
    Returns the best matching title and its score.
    """
    # Get the list of all unique titles from the DataFrame
    all_titles = df_data['title'].tolist()
    
    # Use process.extractOne to find the single best match
    # score_cutoff=70 means we ignore matches below a 70% similarity score
    best_match = process.extractOne(
        user_title, 
        all_titles, 
        scorer=process.fuzz.token_sort_ratio, # Use a good scoring algorithm
        score_cutoff=70
    )
    
    # best_match format is: (Title, Score, Index)
    if best_match:
        # Return only the best matching title if the score is sufficient
        return best_match[0]
    else:
        return None

# --- Flask Routes ---
@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint to get recommendations."""
    if not request.json or 'movie_title' not in request.json:
        return jsonify({"error": "No movie title provided"}), 400

    user_title = request.json['movie_title']
    
    if df is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    # 1. FUZZY MATCHING: Find the exact title from the user's input
    matched_title = find_best_match(user_title, df)

    if not matched_title:
        return jsonify({
            "error": f"Movie title '{user_title}' not found. Please check your spelling or try a common title."
        }), 404

    # 2. CORE AI LOOKUP: Use the perfectly matched title for the lookup
    recommendations = get_recommendations(
        matched_title, cosine_sim, df, indices
    )

    # Note: We now return the title the AI is actually using for clarity
    return jsonify({
        "input_title": user_title,
        "matched_title": matched_title, 
        "recommendations": recommendations
    })

# --- Run the App ---
if __name__ == '__main__':
    # Use 0.0.0.0 for compatibility, debug=True for development
    app.run(host='0.0.0.0', port=5000, debug=True)