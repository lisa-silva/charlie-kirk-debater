import streamlit as st
import requests
import base64
import numpy as np
import io
import wave
import struct

# --- Helper Functions for Audio Processing ---

def base64_to_array_buffer(base64_string):
    """Converts a base64 string to a byte buffer."""
    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        st.error(f"Error decoding base64 string: {e}")
        return None

def pcm_to_wav(pcm_data, sample_rate, channels=1, sample_width=2):
    """Converts raw PCM data to a WAV byte buffer."""
    if not pcm_data:
        return None
        
    # The API returns signed 16-bit PCM data
    try:
        pcm16 = np.frombuffer(pcm_data, dtype=np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width) # 2 bytes = 16 bits
            wav_file.setframerate(sample_rate)
            
            # Pack the PCM data into binary format
            for sample in pcm16:
                wav_file.writeframes(struct.pack('<h', sample))
                
        wav_buffer.seek(0)
        return wav_buffer.read()
        
    except Exception as e:
        st.error(f"Error converting PCM to WAV: {e}")
        st.text("PCM Data Length (bytes): " + str(len(pcm_data)))
        return None

# --- Main App ---

st.set_page_config(layout="centered", page_title="Charlie Kirk Debater")
st.title("🎙️ Charlie Kirk Debater App")
st.write("Enter text and I'll generate an audio clip in a firm, informative debater style. (Powered by Gemini TTS)")

# --- API Configuration ---

# Correct: API key is retrieved from Streamlit secrets
# The user must set this key in their Streamlit Cloud settings.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
    API_KEY = None # App will fail gracefully

# Model is specified in the URL, not the payload
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={API_KEY}"

# --- User Input ---

text_prompt = st.text_area(
    "Enter text to synthesize:", 
    "Let's be very clear, the fundamental argument here is about freedom, and the facts simply don't support the opposing narrative.",
    height=150
)

# Voice configuration (Kore is a good 'firm' voice)
voice_name = "Kore"
style_prompt = "Say in a firm, informative, and confident tone, like a political debater:"

generate_button = st.button("Generate Audio Clip")

if generate_button and API_KEY:
    if not text_prompt:
        st.warning("Please enter some text to generate audio.")
    else:
        with st.spinner("Generating audio... this may take a moment."):
            try:
                # --- API Call ---
                
                # Construct the full prompt for the model
                full_prompt = f"{style_prompt}\n\n{text_prompt}"
                
                # Corrected Payload: No "model" key
                payload = {
                    "contents": [{
                        "parts": [{"text": full_prompt}]
                    }],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": voice_name}
                            }
                        }
                    },
                    # The "model" key is REMOVED from here. It's in the URL.
                }

                headers = {'Content-Type': 'application/json'}

                # Make the API request
                response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
                response.raise_for_status() # Raises an error for bad responses (4xx, 5xx)

                # --- Process Response ---
                result = response.json()
                
                if "candidates" in result and result["candidates"]:
                    part = result["candidates"][0]["content"]["parts"][0]
                    
                    if "inlineData" in part and "data" in part["inlineData"]:
                        audio_data_base64 = part["inlineData"]["data"]
                        mime_type = part["inlineData"]["mimeType"] # e.g., "audio/L16;rate=24000"
                        
                        # Extract sample rate from mime type
                        sample_rate = int(mime_type.split("rate=")[-1])
                        
                        # Convert PCM to WAV
                        pcm_data = base64_to_array_buffer(audio_data_base64)
                        wav_data = pcm_to_wav(pcm_data, sample_rate)
                        
                        if wav_data:
                            st.success("Audio generated successfully!")
                            st.audio(wav_data, format='audio/wav')
                        else:
                            st.error("Failed to convert PCM audio to WAV.")
                            
                    else:
                        st.error("No audio data found in the API response.")
                else:
                    st.error("No candidates found in API response.")
                    st.json(result) # Show the error response

            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP error occurred: {http_err}")
                st.error(f"Response content: {response.text}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

elif generate_button and not API_KEY:
    st.error("Cannot generate audio. The GEMINI_API_KEY is not set in your Streamlit secrets.")
