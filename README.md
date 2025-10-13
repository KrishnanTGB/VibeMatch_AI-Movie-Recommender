# 🎬 VibeMatch: AI Movie Recommender

[![Project Status](https://img.shields.io/badge/Status-Deployed-brightgreen.svg)](https://vibematch-ai-movie-recommender.onrender.com/)
[![Built With](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Backend-Flask%2FGunicorn-lightgray)](https://render.com/)
[![Hosting](https://img.shields.io/badge/Hosted%20On-Render%20Free%20Web%20Service-green)](https://render.com/)
[![Model](https://img.shields.io/badge/Model-Content%20Based%20Filtering-orange.svg)]()

## 🌟 Overview

**VibeMatch** is an intelligent web application designed to eliminate "scrolling fatigue" by providing instant, highly relevant movie recommendations based on a film's **plot and content**.

This project serves as a comprehensive demonstration of full-stack AI development, successfully deploying a large machine learning model on a memory-constrained free cloud environment.

### ✨ Key Features

* **Core AI:** Uses **TF-IDF Vectorization** and **Cosine Similarity** to mathematically measure the semantic closeness between movie plots (Content-Based Filtering).
* **Robust Data:** Combines multiple TMDB datasets for wide coverage, resulting in $\approx 20,000$ unique films before filtering.
* **Memory Optimization (Critical):** The final model is aggressively filtered to the **top 5,000 most popular titles** (`model.pkl`) to ensure it loads successfully within Render's tight **512 MB Free Web Service RAM limit**.
* **Fuzzy Matching:** Implements the `thefuzz` library to tolerate user typos and partial names (e.g., searching "dark knite" successfully finds "The Dark Knight").

---

## 🚀 Live Application

The application is hosted on Render's Free Web Service. Please note that the service **spins down** after 15 minutes of inactivity. The first request after a cold start may take up to 30 seconds to load the model into memory.

🔗 **Access VibeMatch Live:** [**[https://vibematch-ai-movie-recommender.onrender.com/]**](https://vibematch-ai-movie-recommender.onrender.com/)

---

## ⚙️ Technical Architecture & Setup

### Deployment Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend** | HTML / CSS / Vanilla JavaScript | User interface and AJAX handling. |
| **Backend** | Python / Flask | REST API to serve predictions. |
| **Model** | Scikit-learn (TF-IDF, Cosine Similarity) | Core AI prediction logic. |
| **WSGI Server** | Gunicorn | Production server for Flask application. |
| **Hosting** | Render Web Services (Free Tier) | Continuous deployment via GitHub. |

### Project Structure
movie-recommender/
├── data/
│   ├── tmdb_5000_movies.csv
│   └── new_movie_data.csv
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── model.pkl           # <--- The pre-trained 5K movie AI model
├── app.py              # <--- The main Flask application (Gunicorn entry point)
├── model_builder.py    # <--- Script used to build and optimize the model
├── requirements.txt
└── Procfile            # <--- Render/Gunicorn command file

### Local Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KrishnanTGB/VibeMatch_AI-Movie-Recommender
    cd movie-recommender
    ```

2.  **Download Data:** Manually download the two raw CSV files and place them into the `data/` folder.

3.  **Setup Environment & Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate (for Windows .\venv\Scripts\activate)
    pip install -r requirements.txt
    ```

4.  **Build the Optimized Model:** This step generates the memory-friendly `model.pkl`.
    ```bash
    python model_builder.py
    ```

5.  **Run Locally (Development):**
    ```bash
    python app.py
    ```
    (The app will run at `http://127.0.0.1:5000/`)

### Render Deployment Configuration

This project is configured for seamless deployment on Render:

| Render Setting | Value | Purpose |
| :--- | :--- | :--- |
| **Build Command** | `pip install -r requirements.txt && python model_builder.py` | Installs dependencies AND **rebuilds the small model** on the server. |
| **Start Command** | `gunicorn app:app` | Starts the Flask application using Gunicorn. |
| **Root Directory** | (Not specified / Project Root) | Assumes `app.py`, `model.pkl`, and the `templates/` folder are in the top directory or relative paths are set correctly in `app.py`. |

---

## 👤 Author

**Author:** Krishnan T G B

**GitHub:** [@KrishnanTGB]
