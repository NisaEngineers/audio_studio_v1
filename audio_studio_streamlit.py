"""
Audio Studio - Streamlit Application
Complete Audio Manipulation & Transcription Tool

Features:
- Upload and slice audio into multiple segments
- Merge multiple audio files
- Convert between formats (MP3, WAV, OGG, FLAC, M4A)
- Transcribe audio in 15+ languages
- Optimized for 24GB RAM + 4GB GPU
- Uses Whisper 'base' model for efficient transcription
"""

import streamlit as st
import whisper
import torch
import os
import tempfile
import io
from pydub import AudioSegment
import json
from typing import List, Tuple
import base64

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Audio Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER UI
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 8px;
        padding: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

SUPPORTED_FORMATS = ["mp3", "wav", "ogg", "flac", "m4a", "aac"]
WHISPER_MODEL_SIZE = "base"  # Optimized for 4GB GPU

LANGUAGES = {
    "Auto Detect": None,
    "English": "en",
    "Italian": "it",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Dutch": "nl",
    "Polish": "pl"
}

# ============================================================================
# INITIALIZE WHISPER MODEL (CACHED)
# ============================================================================

@st.cache_resource
def load_whisper_model():
    """Load Whisper model (cached to avoid reloading)"""
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, device

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_audio_duration(audio: AudioSegment) -> float:
    """Get audio duration in seconds"""
    return len(audio) / 1000.0

def create_download_link(file_path: str, filename: str) -> str:
    """Create download link for file"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">⬇️ Download {filename}</a>'

# ============================================================================
# AUDIO MANIPULATION FUNCTIONS
# ============================================================================

def slice_audio_equal(uploaded_file, num_slices: int, output_format: str):
    """Slice audio into equal parts"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # Load audio
        audio = AudioSegment.from_file(tmp_path)
        duration_ms = len(audio)
        duration_s = duration_ms / 1000.0
        
        # Calculate slice duration
        slice_duration_ms = duration_ms // num_slices
        
        # Create output directory
        output_dir = tempfile.mkdtemp()
        output_files = []
        
        # Create slices
        for i in range(num_slices):
            start_ms = i * slice_duration_ms
            end_ms = start_ms + slice_duration_ms if i < num_slices - 1 else duration_ms
            
            segment = audio[start_ms:end_ms]
            
            output_filename = f"slice_{i+1}_of_{num_slices}.{output_format}"
            output_path = os.path.join(output_dir, output_filename)
            segment.export(output_path, format=output_format)
            output_files.append((output_path, output_filename))
        
        # Clean up
        os.unlink(tmp_path)
        
        return output_files, f"✅ Successfully sliced into {num_slices} equal parts. Each ~{duration_s/num_slices:.2f}s", None
    
    except Exception as e:
        return [], None, f"❌ Error: {str(e)}"

def slice_audio_custom(uploaded_file, start_time: float, end_time: float, output_format: str):
    """Slice audio with custom start and end times"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # Load audio
        audio = AudioSegment.from_file(tmp_path)
        duration_s = len(audio) / 1000.0
        
        # Validate times
        if start_time < 0 or end_time > duration_s or start_time >= end_time:
            os.unlink(tmp_path)
            return None, None, f"❌ Invalid time range. Audio duration: {duration_s:.2f}s"
        
        # Convert to milliseconds
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # Extract segment
        segment = audio[start_ms:end_ms]
        
        # Save segment
        output_filename = f"slice_{start_time}s_to_{end_time}s.{output_format}"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        segment.export(output_path, format=output_format)
        
        # Clean up
        os.unlink(tmp_path)
        
        return output_path, f"✅ Extracted {start_time}s to {end_time}s ({end_time - start_time:.2f}s)", None
    
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def merge_audio_files(uploaded_files, output_format: str, crossfade_ms: int):
    """Merge multiple audio files"""
    try:
        if not uploaded_files or len(uploaded_files) < 2:
            return None, None, "❌ Please upload at least 2 audio files to merge"
        
        # Save first file and load
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_files[0].name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_files[0].read())
            tmp_path = tmp.name
        
        merged = AudioSegment.from_file(tmp_path)
        os.unlink(tmp_path)
        
        # Merge remaining files
        for uploaded_file in uploaded_files[1:]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            next_audio = AudioSegment.from_file(tmp_path)
            os.unlink(tmp_path)
            
            if crossfade_ms > 0:
                merged = merged.append(next_audio, crossfade=crossfade_ms)
            else:
                merged = merged + next_audio
        
        # Save merged audio
        output_filename = f"merged.{output_format}"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        merged.export(output_path, format=output_format)
        
        duration = len(merged) / 1000.0
        return output_path, f"✅ Merged {len(uploaded_files)} files. Total duration: {duration:.2f}s", None
    
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def convert_audio_format(uploaded_file, output_format: str, bitrate: str):
    """Convert audio to different format"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # Load audio
        audio = AudioSegment.from_file(tmp_path)
        
        # Save in new format
        output_filename = f"converted.{output_format}"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        audio.export(output_path, format=output_format, bitrate=bitrate)
        
        # Get file sizes
        input_size = os.path.getsize(tmp_path) / (1024 * 1024)
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        
        # Clean up
        os.unlink(tmp_path)
        
        return output_path, f"✅ Converted to {output_format.upper()} | Input: {input_size:.2f}MB → Output: {output_size:.2f}MB", None
    
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def transcribe_audio_file(uploaded_file, language: str, model, device):
    """Transcribe audio using Whisper"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # Get language code
        lang_code = LANGUAGES.get(language, None)
        
        # Transcribe
        if lang_code:
            result = model.transcribe(tmp_path, language=lang_code)
        else:
            result = model.transcribe(tmp_path)
        
        # Extract information
        text = result['text']
        detected_lang = result.get('language', 'unknown')
        duration = result.get('duration', 0)
        
        # Clean up
        os.unlink(tmp_path)
        
        return text, detected_lang, duration, None
    
    except Exception as e:
        return None, None, None, f"❌ Error: {str(e)}"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("<h1>🎵 Audio Studio</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Complete Audio Manipulation & Transcription Tool</p>", unsafe_allow_html=True)
    
    # Sidebar - System Info
    with st.sidebar:
        st.markdown("## 🖥️ System Information")
        
        # Load Whisper model
        whisper_model, device = load_whisper_model()
        
        gpu_available = torch.cuda.is_available()
        st.info(f"**GPU Status:** {'✅ Available' if gpu_available else '❌ Not Available'}")
        st.info(f"**Device:** {device.upper()}")
        st.info(f"**Whisper Model:** {WHISPER_MODEL_SIZE.upper()}")
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            st.info(f"**GPU:** {gpu_name}")
            st.info(f"**VRAM:** {gpu_memory:.1f} GB")
        
        st.markdown("---")
        st.markdown("## 📝 Supported Formats")
        st.write(", ".join([fmt.upper() for fmt in SUPPORTED_FORMATS]))
        
        st.markdown("---")
        st.markdown("## 🌍 Supported Languages")
        st.write(", ".join([lang for lang in LANGUAGES.keys() if lang != "Auto Detect"]))
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["✂️ Audio Slicer", "🔗 Audio Merger", "🔄 Format Converter", "📝 Transcription"])
    
    # ========================================================================
    # TAB 1: AUDIO SLICER
    # ========================================================================
    with tab1:
        st.markdown("## ✂️ Audio Slicer")
        st.markdown("Slice audio into multiple segments with equal or custom time ranges")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload Audio")
            slice_file = st.file_uploader("Choose an audio file", type=SUPPORTED_FORMATS, key="slicer_upload")
            
            if slice_file:
                st.audio(slice_file, format=f"audio/{slice_file.name.split('.')[-1]}")
                
                # Get audio duration
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{slice_file.name.split('.')[-1]}") as tmp:
                    tmp.write(slice_file.read())
                    tmp_path = tmp.name
                    audio = AudioSegment.from_file(tmp_path)
                    duration = get_audio_duration(audio)
                    os.unlink(tmp_path)
                    slice_file.seek(0)  # Reset file pointer
                
                st.markdown(f"<div class='info-box'>📊 Audio Duration: {duration:.2f} seconds</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Slice Options")
            slice_mode = st.radio("Slice Mode:", ["Equal Parts", "Custom Range"], horizontal=True)
            
            output_format_slice = st.selectbox("Output Format:", SUPPORTED_FORMATS, key="slice_format")
            
            if slice_mode == "Equal Parts":
                num_slices = st.number_input("Number of Slices:", min_value=1, max_value=100, value=5, step=1)
                
                if st.button("🔪 Slice into Equal Parts", key="slice_equal"):
                    if slice_file:
                        with st.spinner("Slicing audio..."):
                            output_files, success_msg, error_msg = slice_audio_equal(slice_file, num_slices, output_format_slice)
                        
                        if error_msg:
                            st.markdown(f"<div class='error-box'>{error_msg}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='success-box'>{success_msg}</div>", unsafe_allow_html=True)
                            
                            st.markdown("### 📥 Download Sliced Files")
                            for file_path, filename in output_files:
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.markdown(f"**{filename}**")
                                with col_b:
                                    with open(file_path, 'rb') as f:
                                        st.download_button(
                                            label="Download",
                                            data=f,
                                            file_name=filename,
                                            mime=f"audio/{output_format_slice}",
                                            key=f"download_{filename}"
                                        )
                    else:
                        st.warning("⚠️ Please upload an audio file first")
            
            else:  # Custom Range
                start_time = st.number_input("Start Time (seconds):", min_value=0.0, value=0.0, step=0.1)
                end_time = st.number_input("End Time (seconds):", min_value=0.0, value=10.0, step=0.1)
                
                if st.button("🔪 Slice Custom Range", key="slice_custom"):
                    if slice_file:
                        with st.spinner("Slicing audio..."):
                            output_path, success_msg, error_msg = slice_audio_custom(slice_file, start_time, end_time, output_format_slice)
                        
                        if error_msg:
                            st.markdown(f"<div class='error-box'>{error_msg}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='success-box'>{success_msg}</div>", unsafe_allow_html=True)
                            
                            st.markdown("### 📥 Download Sliced File")
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Sliced Audio",
                                    data=f,
                                    file_name=os.path.basename(output_path),
                                    mime=f"audio/{output_format_slice}"
                                )
                    else:
                        st.warning("⚠️ Please upload an audio file first")
    
    # ========================================================================
    # TAB 2: AUDIO MERGER
    # ========================================================================
    with tab2:
        st.markdown("## 🔗 Audio Merger")
        st.markdown("Merge multiple audio files into a single file")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload Audio Files")
            merge_files = st.file_uploader("Choose audio files to merge (multiple)", type=SUPPORTED_FORMATS, accept_multiple_files=True, key="merger_upload")
            
            if merge_files:
                st.markdown(f"<div class='info-box'>📁 {len(merge_files)} files uploaded</div>", unsafe_allow_html=True)
                
                for idx, file in enumerate(merge_files, 1):
                    st.markdown(f"**{idx}. {file.name}**")
        
        with col2:
            st.markdown("### Merge Options")
            output_format_merge = st.selectbox("Output Format:", SUPPORTED_FORMATS, key="merge_format")
            crossfade_ms = st.slider("Crossfade (milliseconds):", 0, 5000, 0, 100)
            
            st.markdown(f"<div class='info-box'>ℹ️ Crossfade creates smooth transitions between files</div>", unsafe_allow_html=True)
            
            if st.button("🔗 Merge Audio Files", key="merge_btn"):
                if merge_files and len(merge_files) >= 2:
                    with st.spinner("Merging audio files..."):
                        output_path, success_msg, error_msg = merge_audio_files(merge_files, output_format_merge, crossfade_ms)
                    
                    if error_msg:
                        st.markdown(f"<div class='error-box'>{error_msg}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='success-box'>{success_msg}</div>", unsafe_allow_html=True)
                        
                        st.markdown("### 🎧 Preview & Download")
                        st.audio(output_path)
                        
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Merged Audio",
                                data=f,
                                file_name=os.path.basename(output_path),
                                mime=f"audio/{output_format_merge}"
                            )
                elif not merge_files:
                    st.warning("⚠️ Please upload audio files first")
                else:
                    st.warning("⚠️ Please upload at least 2 audio files to merge")
    
    # ========================================================================
    # TAB 3: FORMAT CONVERTER
    # ========================================================================
    with tab3:
        st.markdown("## 🔄 Format Converter")
        st.markdown("Convert audio files between different formats")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload Audio")
            convert_file = st.file_uploader("Choose an audio file to convert", type=SUPPORTED_FORMATS, key="converter_upload")
            
            if convert_file:
                st.markdown(f"<div class='info-box'>📄 Current format: {convert_file.name.split('.')[-1].upper()}</div>", unsafe_allow_html=True)
                st.audio(convert_file, format=f"audio/{convert_file.name.split('.')[-1]}")
        
        with col2:
            st.markdown("### Conversion Options")
            output_format_convert = st.selectbox("Output Format:", SUPPORTED_FORMATS, key="convert_format")
            bitrate = st.selectbox("Bitrate:", ["128k", "192k", "256k", "320k"], index=1)
            
            st.markdown(f"<div class='info-box'>ℹ️ Higher bitrate = Better quality but larger file size</div>", unsafe_allow_html=True)
            
            if st.button("🔄 Convert Format", key="convert_btn"):
                if convert_file:
                    with st.spinner("Converting audio format..."):
                        output_path, success_msg, error_msg = convert_audio_format(convert_file, output_format_convert, bitrate)
                    
                    if error_msg:
                        st.markdown(f"<div class='error-box'>{error_msg}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='success-box'>{success_msg}</div>", unsafe_allow_html=True)
                        
                        st.markdown("### 🎧 Preview & Download")
                        st.audio(output_path)
                        
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Converted Audio",
                                data=f,
                                file_name=os.path.basename(output_path),
                                mime=f"audio/{output_format_convert}"
                            )
                else:
                    st.warning("⚠️ Please upload an audio file first")
    
    # ========================================================================
    # TAB 4: TRANSCRIPTION
    # ========================================================================
    with tab4:
        st.markdown("## 📝 Audio Transcription")
        st.markdown("Transcribe audio to text using Whisper AI")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload Audio")
            transcribe_file = st.file_uploader("Choose an audio file to transcribe", type=SUPPORTED_FORMATS, key="transcribe_upload")
            
            if transcribe_file:
                st.audio(transcribe_file, format=f"audio/{transcribe_file.name.split('.')[-1]}")
        
        with col2:
            st.markdown("### Transcription Options")
            language_select = st.selectbox("Language:", list(LANGUAGES.keys()), key="language_select")
            
            st.markdown(f"<div class='info-box'>ℹ️ Auto Detect will identify the language automatically</div>", unsafe_allow_html=True)
            
            if st.button("📝 Transcribe Audio", key="transcribe_btn"):
                if transcribe_file:
                    with st.spinner("Transcribing audio... This may take a moment..."):
                        text, detected_lang, duration, error_msg = transcribe_audio_file(
                            transcribe_file, 
                            language_select, 
                            whisper_model, 
                            device
                        )
                    
                    if error_msg:
                        st.markdown(f"<div class='error-box'>{error_msg}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='success-box'>✅ Transcription completed!</div>", unsafe_allow_html=True)
                        
                        # Display metadata
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Detected Language:** {detected_lang.upper()}")
                        with col_b:
                            st.markdown(f"**Duration:** {duration:.2f} seconds")
                        
                        # Display transcription
                        st.markdown("### 📄 Transcription Result")
                        st.text_area("", value=text, height=300, key="transcription_output")
                        
                        # Download as text file
                        st.download_button(
                            label="⬇️ Download Transcription (TXT)",
                            data=text,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )
                else:
                    st.warning("⚠️ Please upload an audio file first")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #95a5a6;'>Audio Studio v1.0 | Powered by Whisper AI & Streamlit</p>",
        unsafe_allow_html=True
    )

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
