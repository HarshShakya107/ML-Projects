import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (MovieLens or custom)
movies = pd.read_csv(r"C:\\Users\\HP\\OneDrive\\Desktop\\Ml Projects\\movie2.csv")   # contains movieId, title, genres

# Feature engineering
movies['tags'] = movies['genres']  # simple tags from genres
vectorizer = TfidfVectorizer(stop_words="english")
vector_matrix = vectorizer.fit_transform(movies['tags'])

# Similarity matrix
similarity = cosine_similarity(vector_matrix)

# Recommendation function
def recommend(movie):
    if movie not in movies['title'].values:
        return ["Movie not found"]
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    for i in distances[1:6]:  # top 5 recommendations
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.write("Top Recommendations:")
    for movie in recommendations:
        st.write(movie)
