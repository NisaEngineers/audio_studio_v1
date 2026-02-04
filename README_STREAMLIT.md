# Audio Studio - Streamlit Version
## Complete Audio Manipulation & Transcription Application

### 🎯 Features
✅ **Audio Slicer** - Cut audio into equal parts or custom time ranges  
✅ **Audio Merger** - Combine multiple audio files with optional crossfade  
✅ **Format Converter** - Convert between MP3, WAV, OGG, FLAC, M4A, AAC  
✅ **Transcription** - AI-powered speech-to-text in 15+ languages  
✅ **GPU Accelerated** - Optimized for 4GB GPU (Whisper 'base' model)  
✅ **Modern UI** - Clean, intuitive Streamlit interface  

---

## 📋 System Requirements

- **RAM**: 24 GB recommended
- **GPU**: 4 GB VRAM (NVIDIA with CUDA support)
- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or macOS
- **FFmpeg**: Required for audio processing

---

## 🚀 Installation

### Step 1: Install FFmpeg

**Windows:**
```bash
# Using Chocolatey (recommended)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Verify installation:**
```bash
ffmpeg -version
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv audio_env

# Activate virtual environment
# Windows:
audio_env\Scripts\activate

# Linux/macOS:
source audio_env/bin/activate
```

### Step 3: Install Python Dependencies

```bash
# Install all requirements
pip install -r requirements_streamlit.txt

# For CUDA GPU support (NVIDIA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection:
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

### Step 4: Run the Application

```bash
streamlit run audio_studio_streamlit.py
```

The application will automatically open in your default browser at:
```
http://localhost:8501
```

---

## 🎨 User Interface Guide

### Sidebar - System Information
- **GPU Status**: Shows if GPU is available
- **Device**: Indicates CUDA or CPU
- **Whisper Model**: Shows loaded model (base)
- **Supported Formats**: Lists all available audio formats
- **Supported Languages**: Shows available transcription languages

### Main Tabs

#### 1️⃣ Audio Slicer
**Equal Parts Mode:**
- Upload audio file
- Select number of slices (1-100)
- Choose output format
- Click "Slice into Equal Parts"
- Download all sliced files individually

**Custom Range Mode:**
- Upload audio file
- Set start time (seconds)
- Set end time (seconds)
- Choose output format
- Click "Slice Custom Range"
- Download the sliced segment

**Use Cases:**
- Extract highlights from podcasts
- Create samples from songs
- Split interviews into segments
- Prepare training data for voice cloning

#### 2️⃣ Audio Merger
- Upload 2 or more audio files
- Select output format
- Adjust crossfade (0-5000ms) for smooth transitions
- Click "Merge Audio Files"
- Preview and download merged file

**Use Cases:**
- Combine intro + content + outro
- Merge multiple recordings
- Create audio compilations
- Join separated audio tracks

#### 3️⃣ Format Converter
- Upload audio file (any supported format)
- Select target format (MP3, WAV, OGG, FLAC, M4A, AAC)
- Choose bitrate (128k, 192k, 256k, 320k)
- Click "Convert Format"
- Preview and download converted file

**Bitrate Guide:**
- **128k**: Good for voice, small files
- **192k**: Balanced quality/size
- **256k**: High quality music
- **320k**: Maximum quality MP3

**Use Cases:**
- Convert WAV to MP3 for sharing
- Convert to lossless FLAC for archiving
- Optimize file size with different bitrates
- Prepare audio for specific platforms

#### 4️⃣ Transcription
- Upload audio file
- Select language or use "Auto Detect"
- Click "Transcribe Audio"
- View transcription with metadata (language, duration)
- Download transcription as TXT file

**Supported Languages:**
English, Italian, Spanish, French, German, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Turkish, Dutch, Polish

**Use Cases:**
- Create subtitles from video audio
- Transcribe interviews
- Convert lectures to text
- Generate meeting notes
- Accessibility (captions)

---

## 🔧 Performance Optimization

### Whisper Model Information

The application uses **Whisper 'base'** model by default:
- **Size**: ~140 MB
- **VRAM Usage**: 1-2 GB
- **Speed**: Fast on 4GB GPU
- **Accuracy**: Very good for most use cases

### Model Comparison Table

| Model  | Size    | VRAM   | Speed      | Accuracy |
|--------|---------|--------|------------|----------|
| tiny   | 39 MB   | <1 GB  | Very Fast  | Basic    |
| base   | 140 MB  | 1-2 GB | Fast       | Good ✅  |
| small  | 244 MB  | 2-3 GB | Moderate   | Better   |
| medium | 769 MB  | 4-6 GB | Slow       | High     |
| large  | 1550 MB | 8-10GB | Very Slow  | Best     |

**✅ Recommended for 4GB GPU: base or small**

### Changing the Model

To use a different model, edit `audio_studio_streamlit.py`:

```python
WHISPER_MODEL_SIZE = "small"  # Change from "base" to "small", "medium", etc.
```

**Warning**: Models larger than 'small' may cause memory issues on 4GB GPU.

---

## 💡 Tips & Best Practices

### Audio Quality
1. **Upload high-quality source files** for best results
2. Use **lossless formats (WAV, FLAC)** during editing
3. Convert to **compressed formats (MP3)** only for final output
4. Higher bitrate = better quality but larger files

### Transcription Accuracy
1. **Use clean audio** with minimal background noise
2. **Speak clearly** and at moderate pace
3. **Select the correct language** when known
4. For mixed languages, use "Auto Detect"
5. Shorter clips (< 10 minutes) transcribe faster

### File Management
1. **Original files are not modified** - all operations create new files
2. Download files immediately as they're **temporarily stored**
3. Close browser tab to clear temporary files
4. For large batches, process in smaller groups

### Performance Tips
1. **Close unnecessary applications** to free GPU memory
2. **Use shorter audio clips** for faster processing
3. **Transcribe multiple short clips** instead of one long file
4. Monitor GPU usage in Task Manager (Windows) or nvidia-smi (Linux)

---

## 🐛 Troubleshooting

### GPU Not Detected

**Problem**: Application shows "GPU: ❌ Not Available"

**Solutions**:
```bash
# 1. Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# 2. Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Update NVIDIA drivers
# Download from: https://www.nvidia.com/drivers
```

### FFmpeg Not Found

**Problem**: Error about FFmpeg not being found

**Solutions**:
```bash
# Verify FFmpeg installation
ffmpeg -version

# If not found, install FFmpeg (see Installation section)
# Windows: choco install ffmpeg
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# Restart terminal after installation
```

### Out of Memory Error

**Problem**: CUDA out of memory during transcription

**Solutions**:
1. Change model to `tiny` or reduce to CPU:
   ```python
   WHISPER_MODEL_SIZE = "tiny"
   ```
2. Close other GPU-intensive applications
3. Process shorter audio clips
4. Restart the application

### Slow Transcription

**Problem**: Transcription takes too long

**Possible Causes & Solutions**:
- **Using CPU instead of GPU**: Verify GPU is detected in sidebar
- **Large audio file**: Split into smaller chunks first
- **Wrong model**: Ensure using 'base' not 'large'

### Upload Failed

**Problem**: Cannot upload audio file

**Solutions**:
1. Check file format is supported (MP3, WAV, OGG, FLAC, M4A, AAC)
2. Reduce file size (< 200MB recommended)
3. Ensure file is not corrupted
4. Try different browser (Chrome recommended)

### Format Conversion Failed

**Problem**: Error during format conversion

**Solutions**:
1. Verify FFmpeg is installed correctly
2. Check input file is not corrupted
3. Try different output format
4. Ensure enough disk space

---

## 📚 Examples

### Example 1: Extract Podcast Segment
```
1. Go to "Audio Slicer" tab
2. Upload podcast.mp3
3. Select "Custom Range"
4. Start: 120, End: 180 (extracts 2:00 to 3:00)
5. Output Format: wav
6. Click "Slice Custom Range"
7. Download the 60-second segment
```

### Example 2: Create Audiobook from Chapters
```
1. Go to "Audio Merger" tab
2. Upload chapter1.mp3, chapter2.mp3, chapter3.mp3
3. Output Format: mp3
4. Crossfade: 1000ms
5. Click "Merge Audio Files"
6. Download complete audiobook
```

### Example 3: Optimize File Size
```
1. Go to "Format Converter" tab
2. Upload large_file.wav (50MB)
3. Output Format: mp3
4. Bitrate: 192k
5. Click "Convert Format"
6. Download compressed MP3 (5MB)
```

### Example 4: Transcribe Interview
```
1. Go to "Transcription" tab
2. Upload interview.wav
3. Language: English (or Auto Detect)
4. Click "Transcribe Audio"
5. Copy text or download TXT file
```

### Example 5: Slice Song for Samples
```
1. Go to "Audio Slicer" tab
2. Upload song.wav (3 minutes = 180 seconds)
3. Select "Equal Parts"
4. Number of Slices: 12 (each 15 seconds)
5. Output Format: wav
6. Click "Slice into Equal Parts"
7. Download all 12 samples
```

---

## 🔒 Privacy & Security

- **All processing is LOCAL** - files are not sent to external servers
- **Temporary files** are stored locally and deleted when browser closes
- **No data collection** - your audio and transcriptions remain private
- **Offline capable** - works without internet (after initial setup)

---

## 🚀 Advanced Usage

### Network Access
By default, the app runs on localhost only. To access from other devices:

```bash
streamlit run audio_studio_streamlit.py --server.address 0.0.0.0
```

Then access via: `http://YOUR_IP:8501`

### Custom Port
```bash
streamlit run audio_studio_streamlit.py --server.port 8080
```

### Disable GPU (Force CPU)
Edit `audio_studio_streamlit.py`:
```python
# Add at the top
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Batch Processing Workflow

**Process 100 audio files:**
1. Merge files in groups of 10
2. Slice merged files as needed
3. Transcribe each slice
4. Combine transcriptions externally

---

## 📊 Technical Specifications

### Supported Audio Formats
- **Input**: MP3, WAV, OGG, FLAC, M4A, AAC, WMA
- **Output**: MP3, WAV, OGG, FLAC, M4A, AAC

### Audio Quality
- **Sample Rates**: Automatic detection and preservation
- **Bitrates**: 128k, 192k, 256k, 320k (lossy formats)
- **Channels**: Mono and Stereo supported

### File Size Limits
- **Default**: 200 MB per file (Streamlit default)
- **Recommended**: < 100 MB for smooth operation
- **Maximum**: Limited by available RAM

### Processing Speed
| Task                | 4GB GPU    | CPU (i7)   |
|---------------------|------------|------------|
| Slice 1 min audio   | < 1 sec    | < 2 sec    |
| Merge 3 files       | < 2 sec    | < 3 sec    |
| Convert format      | < 3 sec    | < 5 sec    |
| Transcribe 1 min    | 3-5 sec    | 20-30 sec  |

---

## 🆘 Support & Feedback

### Common Questions

**Q: Can I process video files?**  
A: Extract audio from video first using external tools, then process the audio.

**Q: Is there a batch processing mode?**  
A: Not in the UI, but you can upload multiple files to the merger.

**Q: Can I save my settings?**  
A: Settings are session-based. Streamlit doesn't persist state between sessions.

**Q: Does it work offline?**  
A: Yes, after initial model download. No internet required for processing.

**Q: Can I use larger Whisper models?**  
A: Yes, but models larger than 'base' may struggle on 4GB GPU.

### Reporting Issues
- Check GPU is detected in sidebar
- Verify FFmpeg is installed
- Try restarting the application
- Check error messages for specific issues

---

## 📝 Version History

**v1.0** - Initial Release
- Audio slicer (equal & custom)
- Audio merger with crossfade
- Format converter with bitrate selection
- Multi-language transcription
- Streamlit UI with modern design
- GPU acceleration support

---

## 🎓 Learning Resources

- **Whisper Documentation**: https://github.com/openai/whisper
- **Streamlit Docs**: https://docs.streamlit.io
- **Pydub Tutorial**: https://github.com/jiaaro/pydub
- **FFmpeg Guide**: https://ffmpeg.org/documentation.html

---

**Happy Audio Processing! 🎵**
