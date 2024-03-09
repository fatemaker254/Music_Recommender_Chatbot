import base64
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


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("image.jpg")
# adding background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image:url("data:image/jpg;base64,{img}");
background-size: 100%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
[data-testid="sttextArea"] {{
color: white;
}}
[data-testid="sttext_input] {{
color: black; /* Change text color to black */
background-color: white; /* Change background color to white */
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# Initialize session_state if not exists


def creds_entered():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if (
        st.session_state["user"].strip() == "admin"
        and st.session_state["passwd"].strip() == "admin"
    ):
        st.session_state["authenticated"] = True
    else:
        st.session_state["authenticated"] = False
        if not st.session_state["passwd"]:
            st.warning("Please enter a password")
        elif not st.session_state["user"]:
            st.warning("Please enter a username")
        else:
            st.error("The username or password you entered is incorrect")


def authenticate_user():
    if "authenticated" not in st.session_state:
        st.text_input(label="Username", value="", key="user", on_change=creds_entered)
        st.text_input(
            label="Password",
            value="",
            key="passwd",
            type="password",
            on_change=creds_entered,
        )
        return False
    else:
        if st.session_state["authenticated"]:
            return True
        else:
            st.text_input(
                label="Username", value="", key="user", on_change=creds_entered
            )
            st.text_input(
                label="Password",
                value="",
                key="passwd",
                type="password",
                on_change=creds_entered,
            )
            return False


if authenticate_user():
    st.write(
        "<font size='+6'>🤖 : Hi, What song do you want to listen today?</font>",
        unsafe_allow_html=True,
    )
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
        recommendations = df.iloc[song_indices][
            ["Artist", "Track", "Uri", "Duration_ms"]
        ]
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
        capitalized_string = prompt.title()

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            recommendations = recommend_songs(
                capitalized_string, cosine_sim, df, tfidf_vectorizer
            )
            message_content = ""
            if isinstance(recommendations, str):
                message_content = recommendations
            else:
                message_content = "\n".join(
                    [
                        f"- [{rec['Track']} by {rec['Artist']}]({rec['Uri']})"
                        for _, rec in recommendations.iterrows()
                    ]
                )
            st.markdown(message_content, unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": message_content}
        )
