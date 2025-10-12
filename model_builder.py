import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# --- Configuration ---
OLD_DATA_FILE = 'data/tmdb_5000_movies.csv'
NEW_DATA_FILE = 'data/new_movie_data.csv' 
MODEL_FILE = 'model.pkl'
# CRITICAL: We need 'vote_count' now to filter for the best movies
REQUIRED_COLUMNS = ['title', 'overview', 'vote_count'] 

# New constant for the final size target
TARGET_MOVIE_COUNT = 5000 # Keep model size manageable on 512MB RAM

def build_and_save_model():
    # ... (Load Datasets section remains mostly the same, but ensure we load 'vote_count') ...
    try:
        # Load Datasets - Ensure 'vote_count' is in the columns of the new file
        df_old = pd.read_csv(OLD_DATA_FILE, usecols=REQUIRED_COLUMNS)
        df_new = pd.read_csv(NEW_DATA_FILE, usecols=REQUIRED_COLUMNS) 
        
    except FileNotFoundError as e:
        print(f"Error: One of the data files not found: {e}. Check your 'data/' folder.")
        return
    except ValueError:
        print("Error: Required columns ('title', 'overview', 'vote_count') not found. Check your new CSV column names!")
        return

    # 1. Combine and Deduplicate (same as before)
    combined_df = pd.concat([df_new, df_old], ignore_index=True)
    df = combined_df.drop_duplicates(subset=['title'], keep='first')
    df.reset_index(inplace=True, drop=True)

    print(f"Initial unique dataset size: {len(df)}")
    
    # --- STEP 2: CRITICAL SAMPLING FOR MEMORY ---
    
    # Clean 'vote_count' and sort
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
    
    # Keep only the top TARGET_MOVIE_COUNT movies by vote_count
    df_filtered = df.sort_values('vote_count', ascending=False).head(TARGET_MOVIE_COUNT)
    df_filtered.reset_index(inplace=True, drop=True)
    
    df_final = df_filtered[['title', 'overview']] # Keep only what's needed for the model
    
    print(f"Final sampled dataset size for model: {len(df_final)}")
    
    # --- STEP 3: AI Model Building on the smaller data ---
    
    # FIX: Use .loc[] to explicitly modify the DataFrame
    df_final.loc[:, 'overview'] = df_final['overview'].fillna('')
    
    # Create TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    # Use the smaller DataFrame here!
    tfidf_matrix = tfidf.fit_transform(df_final['overview']) 

    # Calculate Cosine Similarity Matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save the dataframe indices for quick lookup by title
    # Use df_final for the indices map
    indices = pd.Series(df_final.index, index=df_final['title']).drop_duplicates()

    # Save the necessary components (using the smaller df_final)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'df': df_final, 'indices': indices, 'cosine_sim': cosine_sim}, f)
        
    print(f"Model successfully built and saved to {MODEL_FILE}")

# Execute the function to create the model file
if __name__ == '__main__':
    build_and_save_model()