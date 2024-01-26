import streamlit as st
import pandas as pd
import joblib

st.title("Music Recommender Chatbot")
USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
# Load the saved model
tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
cosine_sim = joblib.load("cosine_similarity.joblib")

# Load the dataset
df = pd.read_csv("Spotify_Youtube.csv")

# Initialize session_state if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to recommend songs
def recommend_songs(song_name, cosine_sim, df, tfidf_vectorizer):
    matching_songs = df[df["Track"] == song_name]

    if matching_songs.empty:
        return f"No matching song found for '{song_name}'. Please try another song."

    idx = matching_songs.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    song_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[song_indices][["Artist", "Track", "Uri", "Duration_ms"]]
    return recommendations


# Sidebar with a button to delete chat history


# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Main chat interface
if prompt := st.text_input("You:", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        recommendations = recommend_songs(prompt, cosine_sim, df, tfidf_vectorizer)
        if recommendations.empty:
            message_content = (
                f"No matching song found for '{prompt}'. Please try another song."
            )
        else:
            message_content = "\n".join(
                [
                    f"- [{rec['Track']} by {rec['Artist']}]({rec['Uri']})"
                    for _, rec in recommendations.iterrows()
                ]
            )
        st.markdown(message_content, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": message_content})
