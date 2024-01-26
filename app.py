import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

# Load the dataset
df = pd.read_csv("Spotify_Youtube.csv")

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Combine relevant columns into a single string for TF-IDF
df["combined_text"] = df["Artist"] + " " + df["Track"]

# Fit and transform the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_text"])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Save the model and vectorizer to joblib files
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(cosine_sim, "cosine_similarity.joblib")

print("Model saved successfully.")
