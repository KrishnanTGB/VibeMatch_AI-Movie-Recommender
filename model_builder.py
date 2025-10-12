import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# --- Configuration ---
OLD_DATA_FILE = 'data/tmdb_5000_movies.csv'
NEW_DATA_FILE = 'data/new_movie_data.csv' 
MODEL_FILE = 'model.pkl'
# Define the columns we need from the new data (adjust names if needed!)
REQUIRED_COLUMNS = ['title', 'overview']


def build_and_save_model():
    """
    1. Loads, standardizes, and concatenates two datasets with dtype management.
    2. Deduplicates movies, prioritizing the newer record.
    3. Builds and saves the TF-IDF and Cosine Similarity model.
    """
    try:
        # Load Datasets - Use the 'usecols' argument to only load necessary columns
        df_old = pd.read_csv(OLD_DATA_FILE, usecols=REQUIRED_COLUMNS)
        
        # We load the new data only using the required columns. 
        # This implicitly solves the DtypeWarning by ignoring problematic columns.
        df_new = pd.read_csv(NEW_DATA_FILE, usecols=REQUIRED_COLUMNS) 
        
    except FileNotFoundError as e:
        print(f"Error: One of the data files not found: {e}. Check your 'data/' folder.")
        return
    except ValueError:
        # Catch case where column names (title/overview) might be different in the new file
        print("Error: Required columns ('title' and 'overview') might be missing or misspelled in your new dataset.")
        return


    # --- Step 1: Combine and Deduplicate ---
    
    # Vertical Concatenation: Stacking the new data (keep='first') on top of the old data.
    combined_df = pd.concat([df_new, df_old], ignore_index=True)
    
    # Deduplication: Drop duplicates based on 'title', keeping the FIRST occurrence (the newer one)
    df = combined_df.drop_duplicates(subset=['title'], keep='first')
    df.reset_index(inplace=True, drop=True)

    print(f"Combined dataset size: {len(df_old)} (old) + {len(df_new)} (new) -> {len(df)} (final unique)")
    
    # --- Step 2: AI Model Building ---
    
    # FIX: Use .loc[] to explicitly modify the DataFrame and avoid SettingWithCopyWarning
    df.loc[:, 'overview'] = df['overview'].fillna('')
    
    # Create TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    # Calculate Cosine Similarity Matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save the dataframe indices for quick lookup by title
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    # Save the necessary components
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'df': df, 'indices': indices, 'cosine_sim': cosine_sim}, f)
        
    print(f"Model successfully built and saved to {MODEL_FILE}")

# Execute the function to create the model file
if __name__ == '__main__':
    build_and_save_model()