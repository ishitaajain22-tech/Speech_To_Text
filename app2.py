"""
Speech-to-Text Streamlit App
A modern web-based speech recognition tool built with Streamlit.
Perfect for AI internship demonstrations and portfolios.
"""

import streamlit as st
import speech_recognition as sr
import pyaudio
import wave
import tempfile
import os
from pathlib import Path
import time
import io
import numpy as np
from datetime import datetime
import audioop

# Page configuration
st.set_page_config(
    page_title="Speech-to-Text Transcription Tool",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .transcription-result {
        background: linear-gradient(145deg, #667eea, #764ba2);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 1rem 0;
    }
    
    .recording-indicator {
        display: inline-block;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background-color: #ff4b4b;
        margin-right: 10px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

class StreamlitSpeechTranscriber:
    """Speech-to-Text functionality optimized for Streamlit"""
    
    def __init__(self):
        if 'recognizer' not in st.session_state:
            st.session_state.recognizer = sr.Recognizer()
        
        # Initialize session state variables
        if 'transcription_history' not in st.session_state:
            st.session_state.transcription_history = []
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'audio_frames' not in st.session_state:
            st.session_state.audio_frames = []
    
    def transcribe_audio_data(self, audio_data, engine="google"):
        """Transcribe audio data using specified engine"""
        try:
            if engine == "google":
                text = st.session_state.recognizer.recognize_google(audio_data)
            elif engine == "sphinx":
                text = st.session_state.recognizer.recognize_sphinx(audio_data)
            else:
                text = st.session_state.recognizer.recognize_google(audio_data)
            
            return text, True
            
        except sr.UnknownValueError:
            return "Could not understand the audio. Please try speaking more clearly.", False
        except sr.RequestError as e:
            return f"Error with the recognition service: {e}", False
        except Exception as e:
            return f"An unexpected error occurred: {e}", False
    
    def transcribe_uploaded_file(self, uploaded_file, engine="google"):
        """Transcribe uploaded audio file"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Process the audio file
            with sr.AudioFile(tmp_file_path) as source:
                audio_data = st.session_state.recognizer.record(source)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return self.transcribe_audio_data(audio_data, engine)
            
        except Exception as e:
            return f"Error processing file: {e}", False
    
    def add_to_history(self, transcription, source):
        """Add transcription to history"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.transcription_history.append({
            'timestamp': timestamp,
            'transcription': transcription,
            'source': source
        })
    
    def convert_audio_format(self, input_path):
        """Convert audio to WAV format for speech recognition"""
        try:
            # Create output path
            output_path = input_path.replace(".tmp", ".wav")
            
            # Read the input file
            with wave.open(input_path, 'rb') as wav_in:
                # Get parameters
                n_channels = wav_in.getnchannels()
                samp_width = wav_in.getsampwidth()
                framerate = wav_in.getframerate()
                n_frames = wav_in.getnframes()
                frames = wav_in.readframes(n_frames)
                
                # Write to output file with correct parameters for speech recognition
                with wave.open(output_path, 'wb') as wav_out:
                    wav_out.setnchannels(n_channels)
                    wav_out.setsampwidth(samp_width)
                    wav_out.setframerate(framerate)
                    wav_out.writeframes(frames)
            
            return output_path
        except Exception as e:
            st.error(f"Error converting audio: {e}")
            return None
    
    def start_recording(self):
        """Start audio recording"""
        st.session_state.is_recording = True
        st.session_state.audio_frames = []
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Define callback for recording
        def callback(in_data, frame_count, time_info, status):
            if st.session_state.is_recording:
                st.session_state.audio_frames.append(in_data)
                return (in_data, pyaudio.paContinue)
            else:
                return (in_data, pyaudio.paComplete)
        
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            stream_callback=callback
        )
        
        stream.start_stream()
        st.session_state.audio_stream = stream
        st.session_state.audio_p = p
    
    def stop_recording(self):
        """Stop audio recording and return audio data"""
        if st.session_state.is_recording:
            st.session_state.is_recording = False
            
            # Stop and close the stream
            if hasattr(st.session_state, 'audio_stream'):
                st.session_state.audio_stream.stop_stream()
                st.session_state.audio_stream.close()
            
            if hasattr(st.session_state, 'audio_p'):
                st.session_state.audio_p.terminate()
            
            # Save audio to a temporary file
            if st.session_state.audio_frames:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    wf = wave.open(tmp_file.name, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(16000)
                    wf.writeframes(b''.join(st.session_state.audio_frames))
                    wf.close()
                    return tmp_file.name
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🎤 Speech-to-Text Transcription Tool</div>', unsafe_allow_html=True)
    
    # Initialize transcriber
    transcriber = StreamlitSpeechTranscriber()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Engine selection
        engine = st.selectbox(
            "Recognition Engine",
            ["google", "sphinx"],
            help="Google requires internet, Sphinx works offline"
        )
        
        # Language selection (for display purposes)
        language = st.selectbox(
            "Language",
            ["English", "Spanish", "French", "German", "Chinese"],
            help="Currently supports English only. More languages coming soon!"
        )
        
        st.markdown("---")
        
        # About section
        st.header("📋 About")
        st.write("""
        This tool demonstrates speech recognition capabilities using:
        - **Speech Recognition Library**
        - **Real-time audio processing**
        - **Multiple input methods**
        - **Modern web interface**
        """)
        
        # Statistics
        if st.session_state.transcription_history:
            st.header("📊 Statistics")
            st.metric("Total Transcriptions", len(st.session_state.transcription_history))
            
            total_words = sum(len(item['transcription'].split()) 
                            for item in st.session_state.transcription_history)
            st.metric("Total Words Processed", total_words)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Tab interface for different input methods
        tab1, tab2 = st.tabs(["🎙️ Audio Recording", "📁 File Upload"])
        
        with tab1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("Record Audio for Transcription")
            
            # Two recording options: Streamlit's built-in and our custom real-time
            recording_option = st.radio(
                "Recording Method:",
                ["Use Streamlit's audio input"],
                help="Streamlit's audio input is simpler but real-time gives more control"
            )
            
            if recording_option == "Use Streamlit's audio input":
                # Original Streamlit audio input functionality
                audio_bytes = st.audio_input("Record your voice")
                
                if audio_bytes is not None:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    col_rec1, col_rec2 = st.columns(2)
                    
                    with col_rec1:
                        if st.button("🚀 Transcribe Recording", type="primary", use_container_width=True):
                            with st.spinner("Transcribing your audio..."):
                                try:
                                    # Get the raw bytes from the UploadedFile-like object
                                    if hasattr(audio_bytes, 'getvalue'):
                                        audio_data_bytes = audio_bytes.getvalue()
                                    else:
                                        audio_data_bytes = audio_bytes
                                    
                                    # Save temporarily for processing
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                        tmp_file.write(audio_data_bytes)
                                        tmp_file_path = tmp_file.name
                                    
                                    # Convert audio to proper format for speech recognition
                                    wav_file_path = transcriber.convert_audio_format(tmp_file_path)
                                    
                                    if wav_file_path is None:
                                        st.error("❌ Could not process the recorded audio.")
                                        return
                                    
                                    # Process the converted audio file
                                    with sr.AudioFile(wav_file_path) as source:
                                        # Adjust for ambient noise
                                        st.session_state.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                                        audio_data = st.session_state.recognizer.record(source)
                                    
                                    transcription, success = transcriber.transcribe_audio_data(audio_data, engine)
                                    
                                    if success:
                                        st.markdown(f'<div class="transcription-result">📝 <strong>Transcription:</strong><br>{transcription}</div>', unsafe_allow_html=True)
                                        transcriber.add_to_history(transcription, "Audio Recording")
                                        st.success("✅ Transcription completed successfully!")
                                    else:
                                        st.error(f"❌ {transcription}")
                                    
                                    # Clean up temporary files
                                    if os.path.exists(tmp_file_path):
                                        os.unlink(tmp_file_path)
                                    if wav_file_path and os.path.exists(wav_file_path):
                                        os.unlink(wav_file_path)
                                    
                                except Exception as e:
                                    st.error(f"Error processing audio: {e}")
                    
                    with col_rec2:
                        if st.button("🗑️ Clear Recording", use_container_width=True):
                            st.rerun()
            
            else:  # Real-time recording option
                # Recording controls
                col_rec1, col_rec2 = st.columns(2)
                
                with col_rec1:
                    if st.button("⏺️ Start Recording", type="primary", use_container_width=True, 
                               disabled=st.session_state.is_recording):
                        transcriber.start_recording()
                        st.rerun()
                
                with col_rec2:
                    if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True,
                               disabled=not st.session_state.is_recording):
                        audio_file = transcriber.stop_recording()
                        
                        if audio_file:
                            with st.spinner("Processing your recording..."):
                                try:
                                    # Process the recorded audio file
                                    with sr.AudioFile(audio_file) as source:
                                        # Adjust for ambient noise
                                        st.session_state.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                                        audio_data = st.session_state.recognizer.record(source)
                                    
                                    transcription, success = transcriber.transcribe_audio_data(audio_data, engine)
                                    
                                    if success:
                                        st.markdown(f'<div class="transcription-result">📝 <strong>Transcription:</strong><br>{transcription}</div>', unsafe_allow_html=True)
                                        transcriber.add_to_history(transcription, "Audio Recording")
                                        st.success("✅ Transcription completed successfully!")
                                    else:
                                        st.error(f"❌ {transcription}")
                                    
                                    # Clean up temporary file
                                    if os.path.exists(audio_file):
                                        os.unlink(audio_file)
                                    
                                except Exception as e:
                                    st.error(f"Error processing audio: {e}")
                        
                        st.rerun()
                
                # Show recording status
                if st.session_state.is_recording:
                    st.markdown('<div class="recording-indicator"></div><span style="color: #ff4b4b; font-weight: bold;">Recording in progress... Speak now</span>', unsafe_allow_html=True)
                    st.warning("Click 'Stop Recording' when you're finished speaking.")
                else:
                    st.info("Click 'Start Recording' to begin capturing audio.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("Upload Audio File")
            
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac', 'aiff'],
                help="Supported formats: WAV, MP3, FLAC, AIFF"
            )
            
            if uploaded_file is not None:
                st.audio(uploaded_file)
                
                col_up1, col_up2 = st.columns(2)
                
                with col_up1:
                    st.info(f"📎 **File:** {uploaded_file.name}")
                    st.info(f"📊 **Size:** {uploaded_file.size:,} bytes")
                
                with col_up2:
                    if st.button("🚀 Transcribe File", type="primary", use_container_width=True):
                        with st.spinner("Processing uploaded file..."):
                            transcription, success = transcriber.transcribe_uploaded_file(uploaded_file, engine)
                            
                            if success:
                                st.markdown(f'<div class="transcription-result">📝 <strong>Transcription:</strong><br>{transcription}</div>', unsafe_allow_html=True)
                                transcriber.add_to_history(transcription, f"File: {uploaded_file.name}")
                                st.success("✅ File transcribed successfully!")
                            else:
                                st.error(f"❌ {transcription}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        # History and results panel
        st.subheader("📜 Transcription History")
        
        if st.session_state.transcription_history:
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.transcription_history = []
                st.rerun()
            
            # Display history
            for i, item in enumerate(reversed(st.session_state.transcription_history)):
                with st.expander(f"#{len(st.session_state.transcription_history)-i} - {item['timestamp']}", expanded=(i==0)):
                    st.write(f"**Source:** {item['source']}")
                    st.write(f"**Text:** {item['transcription']}")
                    
                    # Copy to clipboard functionality
                    st.code(item['transcription'])
        else:
            st.info("👋 No transcriptions yet. Try recording some audio or uploading a file!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        🚀 Built with Streamlit for AI Internship Portfolio<br>
        💡 Demonstrates: Speech Recognition • Audio Processing • Web Development • AI Integration
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()