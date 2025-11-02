import base64
import streamlit as st
import requests
import numpy as np
import io
import time
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Vocal Premise Debater",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Constants ---
API_KEY = st.secrets["tool_auth"]["gemini_api_key"]
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta2/models/gemini-2.5-flash-preview-tts:generateContent"

API_KEY = st.secrets["tool_auth"]["AIzaSyDNB-9Z\_Liw2CeFwB-TnkBrplPeBP\_HEPc"]
[tool_auth]
gemini_api_key = "AIzaSyDNB-9Z\_Liw2CeFwB-TnkBrplPeBP\_HEPc"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"

# --- Audio Utility Function (PCM to WAV) ---
def pcm_to_wav(pcm_data, sample_rate=24000):
    """Converts raw 16-bit signed PCM audio data to a WAV file byte array."""
    
    # API returns 16-bit signed linear PCM.
    pcm16 = np.frombuffer(pcm_data, dtype=np.int16)
    
    # 44-byte WAV header size
    buffer = np.zeros(44 + pcm16.size * 2, dtype=np.uint8) 

    # --- RIFF Chunk ---
    buffer[0:4] = b'RIFF'  # ChunkID
    buffer[4:8] = np.uint32(36 + pcm16.size * 2).tobytes() # ChunkSize
    buffer[8:12] = b'WAVE' # Format
    
    # --- FMT Chunk ---
    buffer[12:16] = b'fmt ' # Subchunk1ID
    buffer[16:20] = np.uint32(16).tobytes() # Subchunk1Size (16 for PCM)
    buffer[20:22] = np.uint16(1).tobytes() # AudioFormat (1 for PCM)
    buffer[22:24] = np.uint16(1).tobytes() # NumChannels (Mono)
    buffer[24:28] = np.uint32(sample_rate).tobytes() # SampleRate
    buffer[28:32] = np.uint32(sample_rate * 2).tobytes() # ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    buffer[32:34] = np.uint16(2).tobytes() # BlockAlign (NumChannels * BitsPerSample/8)
    buffer[34:36] = np.uint16(16).tobytes() # BitsPerSample
    
    # --- Data Chunk ---
    buffer[36:40] = b'data' # Subchunk2ID
    buffer[40:44] = np.uint32(pcm16.size * 2).tobytes() # Subchunk2Size
    
    # Copy PCM data after header
    buffer[44:] = pcm16.tobytes()
    
    return io.BytesIO(buffer.tobytes())


# --- API Interaction Functions ---

@st.cache_data
def generate_tts_audio(text_to_speak, voice_name="Kore"):
    """Calls the TTS API and returns a WAV file byte stream."""
    st.info("Generating audio response...", icon="🔊")
    
    headers = {
        'Content-Type': 'application/json',
    }

    payload = {
        "contents": [{
            "parts": [{ "text": text_to_speak }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": { "voiceName": voice_name }
                }
            }
        },
        "model": "gemini-2.5-flash-preview-tts"
    }

    # API call with Exponential Backoff
    for i in range(3): 
        try:
            response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()
            
            candidate = result.get('candidates', [{}])[0]
            part = candidate.get('content', {}).get('parts', [{}])[0]
            
            if part and part.get('inlineData'):
                audio_data_base64 = part['inlineData']['data']
                audio_data_bytes = io.BytesIO(base64.b64decode(audio_data_base64)).read()
                
                # Convert raw PCM audio data to WAV format
                wav_file = pcm_to_wav(audio_data_bytes)
                return wav_file
            
            st.error("Error: TTS response did not contain valid audio data.")
            return None

        except requests.exceptions.RequestException as e:
            if i < 2:
                st.warning(f"API call failed, retrying in {2**i}s... Error: {e}")
                time.sleep(2**i)
            else:
                st.error(f"Final API call failed after retries: {e}")
                return None
        except Exception as e:
            st.error(f"An unexpected error occurred during audio generation: {e}")
            return None
    return None

@st.cache_data
def generate_debate_response(premise, persona="Charlie Kirk"):
    """Calls the Gemini API to generate the debate response."""
    
    system_prompt = f"""You are a vocal debater known for {persona}'s style. 
    Analyze the user's premise. Then, provide a short (2-3 sentence), highly confident, and direct counter-argument or defense of the opposite position, written in the voice of {persona}. 
    Conclude your response with a summary statement.
    """
    
    user_query = f"Analyze and respond to this premise: '{premise}'"

    # API call with Exponential Backoff
    for i in range(3): 
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}",
                headers={'Content-Type': 'application/json'},
                data=json.dumps({
                    "contents": [{"parts": [{"text": user_query}]}],
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "tools": [{"google_search": {}}]
                })
            )
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']

        except requests.exceptions.RequestException as e:
            if i < 2:
                st.warning(f"Text generation failed, retrying in {2**i}s... Error: {e}")
                time.sleep(2**i)
            else:
                st.error(f"Final text generation failed after retries: {e}")
                return "The debate generator failed to connect to the API."
        except Exception as e:
            st.error(f"An unexpected error occurred during debate generation: {e}")
            return "An unexpected error occurred."
    return "The debate generator failed."


# --- Main Streamlit UI ---

st.title("🗣️ Vocal Premise Debater")
st.markdown("Enter a premise, and receive a counter-argument in the style of Charlie Kirk, complete with generated audio.")

# Input
premise = st.text_area(
    "Enter your premise (e.g., 'The voting age should be lowered to 16.')",
    max_chars=200,
    height=100
)

if st.button("Generate Debate", type="primary") and premise:
    if not API_KEY:
        st.error("API Key not found. Please ensure `gemini_api_key` is set in your `.streamlit/secrets.toml` file.")
    else:
        with st.spinner("Generating counter-argument..."):
            
            # 1. Generate Debate Text
            debate_text = generate_debate_response(premise)
            
            # 2. Display Debate Text
            st.subheader("Charlie Kirk's Response:")
            st.info(debate_text)
            
            # 3. Generate and Display Audio
            # We use the 'Kore' voice as a good fit for this persona
            audio_wav_io = generate_tts_audio(debate_text, voice_name="Kore")
            
            if audio_wav_io:
                st.subheader("Listen to the Argument:")
                # Streamlit's audio player accepts the raw WAV data
                st.audio(audio_wav_io.getvalue(), format="audio/wav")

elif st.button("Generate Debate", disabled=True):
    st.warning("Please enter a premise above to start the debate.")

import base64
import streamlit as st
import requests
import numpy as np
import io
import time
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Vocal Premise Debater",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Constants ---
# Use the correct key structure: st.secrets.<section>.<key>
API_KEY = st.secrets.tool_auth.gemini_api_key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"

# --- Audio Utility Function (PCM to WAV) ---
def pcm_to_wav(pcm_data, sample_rate=24000):
    """Converts raw 16-bit signed PCM audio data to a WAV file byte array."""
    
    # API returns 16-bit signed linear PCM.
    pcm16 = np.frombuffer(pcm_data, dtype=np.int16)
    
    # 44-byte WAV header size
    buffer = np.zeros(44 + pcm16.size * 2, dtype=np.uint8) 

    # --- RIFF Chunk ---
    buffer[0:4] = b'RIFF'  # ChunkID
    buffer[4:8] = np.uint32(36 + pcm16.size * 2).tobytes() # ChunkSize
    buffer[8:12] = b'WAVE' # Format
    
    # --- FMT Chunk ---
    buffer[12:16] = b'fmt ' # Subchunk1ID
    buffer[16:20] = np.uint32(16).tobytes() # Subchunk1Size (16 for PCM)
    buffer[20:22] = np.uint16(1).tobytes() # AudioFormat (1 for PCM)
    buffer[22:24] = np.uint16(1).tobytes() # NumChannels (Mono)
    buffer[24:28] = np.uint32(sample_rate).tobytes() # SampleRate
    buffer[28:32] = np.uint32(sample_rate * 2).tobytes() # ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    buffer[32:34] = np.uint16(2).tobytes() # BlockAlign (NumChannels * BitsPerSample/8)
    buffer[34:36] = np.uint16(16).tobytes() # BitsPerSample
    
    # --- Data Chunk ---
    buffer[36:40] = b'data' # Subchunk2ID
    buffer[40:44] = np.uint32(pcm16.size * 2).tobytes() # Subchunk2Size
    
    # Copy PCM data after header
    buffer[44:] = pcm16.tobytes()
    
    return io.BytesIO(buffer.tobytes())


# --- API Interaction Functions ---

@st.cache_data
def generate_tts_audio(text_to_speak, voice_name="Kore"):
    """Calls the TTS API and returns a WAV file byte stream."""
    st.info("Generating audio response...", icon="🔊")
    
    headers = {
        'Content-Type': 'application/json',
    }

    payload = {
        "contents": [{
            "parts": [{ "text": text_to_speak }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": { "voiceName": voice_name }
                }
            }
        },
        "model": "gemini-2.5-flash-preview-tts"
    }

    # API call with Exponential Backoff
    for i in range(3): 
        try:
            response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload))
            response.raise_for_status()

            result = response.json()
            
            candidate = result.get('candidates', [{}])[0]
            part = candidate.get('content', {}).get('parts', [{}])[0]
            
            if part and part.get('inlineData'):
                audio_data_base64 = part['inlineData']['data']
                audio_data_bytes = io.BytesIO(base64.b64decode(audio_data_base64)).read()
                
                # Convert raw PCM audio data to WAV format
                wav_file = pcm_to_wav(audio_data_bytes)
                return wav_file
            
            st.error("Error: TTS response did not contain valid audio data.")
            return None

        except requests.exceptions.RequestException as e:
            if i < 2:
                st.warning(f"API call failed, retrying in {2**i}s... Error: {e}")
                time.sleep(2**i)
            else:
                st.error(f"Final API call failed after retries: {e}")
                return None
        except Exception as e:
            st.error(f"An unexpected error occurred during audio generation: {e}")
            return None
    return None

@st.cache_data
def generate_debate_response(premise, persona="Charlie Kirk"):
    """Calls the Gemini API to generate the debate response."""
    
    system_prompt = f"""You are a vocal debater known for {persona}'s style. 
    Analyze the user's premise. Then, provide a short (2-3 sentence), highly confident, and direct counter-argument or defense of the opposite position, written in the voice of {persona}. 
    Conclude your response with a summary statement.
    """
    
    user_query = f"Analyze and respond to this premise: '{premise}'"

    # API call with Exponential Backoff
    for i in range(3): 
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}",
                headers={'Content-Type': 'application/json'},
                data=json.dumps({
                    "contents": [{"parts": [{"text": user_query}]}],
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "tools": [{"google_search": {}}]
                })
            )
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']

        except requests.exceptions.RequestException as e:
            if i < 2:
                st.warning(f"Text generation failed, retrying in {2**i}s... Error: {e}")
                time.sleep(2**i)
            else:
                st.error(f"Final text generation failed after retries: {e}")
                return "The debate generator failed to connect to the API."
        except Exception as e:
            st.error(f"An unexpected error occurred during debate generation: {e}")
            return "An unexpected error occurred."
    return "The debate generator failed."


# --- Main Streamlit UI ---

st.title("🗣️ Vocal Premise Debater")
st.markdown("Enter a premise, and receive a counter-argument in the style of Charlie Kirk, complete with generated audio.")

# Input
premise = st.text_area(
    "Enter your premise (e.g., 'The voting age should be lowered to 16.')",
    max_chars=200,
    height=100
)

if st.button("Generate Debate", type="primary") and premise:
    if not API_KEY:
        st.error("API Key not found. Please ensure `gemini_api_key` is set in your `.streamlit/secrets.toml` file.")
    else:
        with st.spinner("Generating counter-argument..."):
            
            # 1. Generate Debate Text
            debate_text = generate_debate_response(premise)
            
            # 2. Display Debate Text
            st.subheader("Charlie Kirk's Response:")
            st.info(debate_text)
            
            # 3. Generate and Display Audio
            # We use the 'Kore' voice as a good fit for this persona
            audio_wav_io = generate_tts_audio(debate_text, voice_name="Kore")
            
            if audio_wav_io:
                st.subheader("Listen to the Argument:")
                # Streamlit's audio player accepts the raw WAV data
                st.audio(audio_wav_io.getvalue(), format="audio/wav")

elif st.button("Generate Debate", disabled=True):
    st.warning("Please enter a premise above to start the debate.")
