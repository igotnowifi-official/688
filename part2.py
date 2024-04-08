import requests
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import os
import gradio as gr

#token = os.environ['apikey']
API_HOST = "https://api.voicemod.net"

# Function to generate audio using the Voicemod API
def generate_audio(lyrics):
    url = API_HOST + "/v2/cloud/partners/ttsing"
    headers = {'x-api-key': 'controlapi-q3id32501'}
    payload = {
        # Add payload data here (artist name, song name, lyrics, etc.)
        'lyrics': lyrics
    }
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()
    audio_url = response_data.get('transformedAudioUrl')
    if audio_url:
        # Download the audio file
        audio_file = download_file(audio_url)
        return audio_file
    else:
        return None

# Function to download a file from a URL
def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            return tmp_file.name
    else:
        print("Error: Unable to download file")

# Function to play the generated audio
def play_audio(audio_file):
    if audio_file:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(audio_file)
        # Play the audio
        play(audio)
    else:
        print("Error: Unable to play audio")

# After generating the lyrics and getting the predictions
# Generate audio using the Voicemod API
lyrics = "Your generated lyrics go here"
audio_file = generate_audio(lyrics)
# Play the generated audio
play_audio(audio_file)
