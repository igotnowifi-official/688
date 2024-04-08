# Genius api load
# Clean lyrics
# Visuals of sentiment analysis and metrics of song

import streamlit as st
import lyricsgenius
from datasets import load_dataset
from afinn import Afinn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import datetime
import random
import torch
from part2 import generate_audio, play_audio

# Initialize the Genius API client with your API key
genius = lyricsgenius.Genius('dLgle08YVcWmSdFppd5ZB4A-RvEW4WgA4OdY_JTBogchASwXIvXa9Uo6Y2LETbBB')

# Get the current time
current_time = datetime.datetime.now()

# Format the time as "it's [time] am/pm"
formatted_time = current_time.strftime("it's %I %p, and ")



# Function to retrieve lyrics for a given song
def get_lyrics(song_title, artist_name):
    try:
        # Search for the song
        song = genius.search_song(song_title, artist_name)

        # Retrieve the lyrics
        if song:
            return song.lyrics
        else:
            return None
    except Exception as e:
        print(f"Error retrieving lyrics: {e}")
        return None

def preprocess_lyrics(lyrics):
    # Remove numbers and words within square brackets
    cleaned_lyrics = re.sub(r'\[.*?\]|\d+', '', lyrics)

    # Remove the line containing 'Get tickets as low as $You might also like'
    cleaned_lyrics = re.sub(r'^.*Get tickets as low as \$You might also like.*$', '', cleaned_lyrics, flags=re.MULTILINE)

    # Remove the last occurrence of the word 'Embed' at the end of the lyrics
    cleaned_lyrics = re.sub(r'Embed\s*$', '', cleaned_lyrics.strip())

    # Remove all text before the first occurrence of the word 'Lyrics', including 'Lyrics'
    cleaned_lyrics = re.sub(r'^.*?Lyrics', 'Lyrics', cleaned_lyrics)

    return cleaned_lyrics.strip()

# Function to perform sentiment analysis on the lyrics using Afinn
def analyze_sentiment(lyrics):
    afinn = Afinn()
    sentiment_score = afinn.score(lyrics)
    return sentiment_score

def remove_punctuation(s):
    return re.sub(r'[^\w\s]', '', s)

def extract_keywords(text, n_keywords=5):
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

    # Fit and transform the text
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])

    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get indices of top N keywords
    top_indices = tfidf_matrix.toarray().argsort()[0][-n_keywords:][::-1]

    # Get top N keywords
    top_keywords = [feature_names[i] for i in top_indices]

    return top_keywords

st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6;  /* Light blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Song Lyrics Sentiment Analysis")

# User input for song title and artist
song_title = st.sidebar.text_input("Enter the song title:")
artist_name = st.sidebar.text_input("Enter the artist name:")
st.sidebar.write("Setting for your alternate lyrics:")
start = st.sidebar.text_input("Enter starting sentence:", "I am")
num_sequences = st.sidebar.number_input("Amount of generated texts:", value=3, step=1)
min_length = st.sidebar.number_input("Minimum length:", value=100, step=1)
max_length = st.sidebar.number_input("Maximum length:", value=160, step=1)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=3.0, step=0.01, value=1.0)
top_p = st.sidebar.slider("Top p", min_value=0.0, max_value=1.0, step=0.01, value=0.95)
top_k = st.sidebar.number_input("Top k:", value=50, step=1)
repetition_penalty = st.sidebar.number_input("Repetition penalty:", value=1.0, step=0.1)

# Load the model based on artist name
if artist_name:
    # Convert artist name to lowercase and replace spaces with hyphens
    formatted_artist_name = artist_name.lower().replace(" ", "-")

    model_name = f"huggingartists/{formatted_artist_name}"
    try:
        dataset = load_dataset(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        st.error("Error loading model for generated lyrics. Please check the artist name or try another artist.")
else:
    st.warning("Please enter the artist name.")


# Get Lyrics & Analyze button
if st.sidebar.button("Get Lyrics & Analyze"):
    if song_title and artist_name:

        # Retrieve lyrics for the input song
        lyrics = get_lyrics(song_title, artist_name)

        if lyrics:
            # Preprocess the lyrics by removing unwanted sections
            cleaned_lyrics = preprocess_lyrics(lyrics)

            # Perform sentiment analysis on the cleaned lyrics
            sentiment_score = analyze_sentiment(cleaned_lyrics)

            # Label the sentiment score as positive, negative, or neutral
            if sentiment_score > 0:
                sentiment_label = "Positive"
            elif sentiment_score < 0:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

            # Add CSS style for the pattern border
            st.markdown(
                """
                <style>
                    .pattern-border {
                        border: 5px dashed transparent;
                        border-image: repeating-linear-gradient(
                            45deg,
                            #e91e63,
                            #e91e63 10px,
                            #2196f3 10px,
                            #2196f3 20px
                        );
                        border-image-slice: 1;
                        padding: 20px;
                        border-radius: 10px;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Display the app result with the pattern border
            with st.container():

                # Display cleaned lyrics, sentiment analysis results, and recommendation
                st.subheader("Lyrics:")
                st.text_area(" ", cleaned_lyrics)

                st.subheader(f"Analyzing sentiment for {song_title} by {artist_name}")
                st.success(f"Sentiment Score: {sentiment_score} {sentiment_label}")


                keywords = extract_keywords(lyrics)
                st.success(f"Top Keywords: {keywords}")

                # Generate lyrics
                if model:
                    st.subheader("Alternate Generated Lyrics:")
                    #start = formatted_time  # Use formatted time
                    #num_sequences = random.randint(2, 5)
                    #min_length = random.randint(100, 150)
                    #max_length = random.randint(160, 250)
                    #sentiment_score = random.uniform(-1, 1)
                    #temperature = 0.1 if sentiment_score < 0 else (1.5 if sentiment_score == 0 else 3.0)
                    #top_p = random.random()
                    #top_k = random.randint(25, 50)
                    #repetition_penalty = random.uniform(1, 5)

                    encoded_prompt = tokenizer(start, add_special_tokens=False, return_tensors="pt").input_ids

                    # Generate attention mask
                    attention_mask = torch.ones_like(encoded_prompt)

                    # Set padding tokens to 0
                    attention_mask[encoded_prompt == tokenizer.pad_token_id] = 0

                    # prediction
                    output_sequences = model.generate(
                        input_ids=encoded_prompt,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        min_length=min_length,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k),
                        do_sample=True,
                        repetition_penalty=repetition_penalty,
                        num_return_sequences=num_sequences
                    )

                    # Post-processing
                    predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in output_sequences]

                    # Display generated lyrics
                    for idx, prediction in enumerate(predictions):
                        prediction = re.sub(r'\[.*?\]|\d+', '', prediction)
                        prediction = re.sub(r'(?<!\bI\b)(?<!\.\s)(?<!\.\n)(?<!\.\s\n)(?<![.!?])\s*([A-Z][a-z]*)\b', r'. \1', prediction)
                        st.markdown(f"Generated Text {idx + 1}:<br> > {prediction}", unsafe_allow_html=True)
                        # Generate audio from predicted lyrics
                        #audio_file = generate_audio(prediction)

                        # Play the generated audio
                        #play_audio(audio_file)

                else:
                    st.warning("Model not loaded. Please enter a valid artist name.")

                # Add footer
                st.markdown("---")
                st.markdown("Created by Natasya Liew")

                # Apply custom CSS
                st.markdown(
                    """
                    <style>
                    body {
                        background-color: #f0f2f6;
                        color: #333;
                    }
                    .st-bq {
                        font-size: 18px;
                    }
                    .st-ca {
                        color: #0077cc;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )





        else:
            st.error("Error retrieving lyrics. Please check the song title and artist name.")
    else:
        st.warning("Please enter the song title and artist name.")

