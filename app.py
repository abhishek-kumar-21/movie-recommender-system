import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Movie Recommender", layout="wide")


@st.cache_resource
def load_data():
    with open('movies_data.pkl', 'rb') as f:
        movie_data = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('movie_indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    df = pd.DataFrame(movie_data)
    return df, tfidf_matrix, indices


movies_df, tfidf_matrix, indices = load_data()


def get_recommendations(title, top_n=5):
    if title not in indices:
        return []

    idx = indices[title]
    # Compute similarity on the fly for just this one movie vector
    query_vector = tfidf_matrix[idx]
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Sort and pick top matches (ignoring the movie itself at index 0)
    top_indices = sim_scores.argsort()[::-1][1:top_n + 1]

    return movies_df.iloc[top_indices].to_dict(orient='records')


# UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Search or select a movie to get recommendations:",
    movies_df['title'].values
)

if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_movie, top_n=5)

    if recommendations:
        st.subheader("Recommended Movies")
        cols = st.columns(5)

        # Fallback image if TMDB poster is missing or broken
        placeholder_img = "https://via.placeholder.com/500x750?text=No+Poster"

        for col, rec in zip(cols, recommendations):
            poster_url = rec['poster'] if rec['poster'] and str(rec['poster']).startswith("http") else placeholder_img
            with col:
                st.image(poster_url, use_container_width=True)
                st.caption(f"**{rec['title']}**")
    else:
        st.warning("No recommendations found.")
