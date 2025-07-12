# StreamSnap

![StreamSnap Banner](https://via.placeholder.com/800x200.png?text=StreamSnap+Banner) <!-- Replace with actual banner image if available -->

**StreamSnap** is a powerful, AI-powered desktop application for downloading, transcribing, and analyzing YouTube videos and audio. Built with Python and leveraging cutting-edge libraries like Whisper, YOLO11, and Gradio, StreamSnap provides an intuitive interface for content creators, researchers, and enthusiasts to process media with advanced features like real-time object detection, automatic chapter generation, and interactive transcripts.

## Features

- **YouTube Downloader (StreamDL)**:
  - Download YouTube videos or audio in various formats (MP4, MKV, MP3, WAV, etc.).
  - Support for multiple quality options (1080p, 720p, etc.).
  - GPU-accelerated downloading and processing for faster performance.
  - Download history with thumbnail previews and playback.

- **Intelligent Scribe**:
  - AI-powered transcription using OpenAI's Whisper model (tiny, base, small, medium, large).
  - Interactive transcript with clickable timestamps for easy navigation.
  - Export transcripts as SRT or WebVTT subtitle files.
  - Searchable transcripts with highlighted results.
  - Clip creation from selected transcript segments.

- **Vision Lab**:
  - Real-time object detection using YOLO11 (nano, small, medium, large, extra-large models).
  - Comprehensive video analysis including scene detection, frame quality assessment, and silence detection.
  - Exportable analysis data in JSON format.

- **Performance Optimization**:
  - GPU acceleration for video processing (NVIDIA, Intel, AMD) with CPU fallback.
  - Optimized FFmpeg parameters for efficient encoding/decoding.
  - Smart confidence filtering for improved YOLO11 detection accuracy.

## Prerequisites

- **Python**: Version 3.10.
- **Gradio**: 5.35.0.
- **FFmpeg**: Required for video and audio processing. Install it and add it to your system's PATH.
- **GPU (Optional)**: NVIDIA, Intel, or AMD GPU for accelerated processing (CUDA, Quick Sync, or AMF support).
- **Operating System**: Windows, macOS, or Linux.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/streamsnap.git
   cd streamsnap
   ```

2. **Create a Virtual Environment** (recommended):
   Create a virtual environment to manage dependencies. Use the appropriate command for your operating system:
   ```bash
   python -m venv venv
   ```
   Activate the virtual environment:
   - **On macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - **On Windows**:
     ```bash
     venv\Scripts\activate
     ```

3. **Install Dependencies**:
   Run:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: Ensure `ffmpeg` is installed and accessible in your system's PATH. On Windows, you may need to download it from [FFmpeg's official site](https://ffmpeg.org/download.html). On Linux/macOS, install via package managers:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   # macOS (with Homebrew)
   brew install ffmpeg
   ```

4. **Verify YOLO11 Installation** (optional, for Vision Lab):
   If using the Vision Lab, ensure `ultralytics` is installed correctly:
   ```bash
   pip install ultralytics
   ```

## Usage

1. **Run the Application**:
   Start StreamSnap by running the main script:
   ```bash
   python app.py
   ```

   This launches a web-based Gradio interface in your default browser (typically at `http://localhost:7860`).

2. **StreamDL Tab**:
   - Paste a YouTube URL and click **Preview** to view video details.
   - Configure download settings (audio-only, video quality, format, GPU acceleration).
   - Click **Download Now** to save the media to the `downloads/` folder.
   - Browse and play downloaded media in the **Download History** section.

3. **Intelligent Scribe Tab**:
   - Select a downloaded video from the dropdown.
   - Choose a Whisper model size (e.g., `base` for balance, `large` for accuracy).
   - Click **Transcribe Video** to generate an interactive transcript.
   - Use features like searching, exporting subtitles (SRT/VTT), or creating clips from selected segments.

4. **Vision Lab Tab**:
   - Select a video and a YOLO11 model size.
   - Adjust the confidence threshold (0.25 recommended) and enable smart detection filtering.
   - Click **Start Detection** to begin real-time object detection.

## Project Structure

```
streamsnap/
├── app.py               # Main application script
├── requirements.txt     # Python dependencies
├── downloads/          # Folder for downloaded media
├── clips/              # Folder for generated clips
└── download_history.json # Stores download history
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate tests.

## Troubleshooting

- **FFmpeg Not Found**: Ensure FFmpeg is installed and added to your system's PATH.
- **YOLO11 Detection Issues**: Verify `ultralytics` is installed and try the "Test YOLO" button to diagnose model issues.
- **GPU Acceleration Not Working**: Check GPU drivers and compatibility (CUDA for NVIDIA, Quick Sync for Intel, AMF for AMD).
- **Transcription Errors**: Ensure the selected video has audio and try a smaller Whisper model for faster processing.

For additional help, open an issue on the [GitHub repository](https://github.com/your-username/streamsnap/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Powered by [Gradio](https://gradio.app/) for the web interface.
- [Whisper](https://github.com/openai/whisper) for transcription.
- [YOLO11](https://github.com/ultralytics/ultralytics) for object detection.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube downloading.
- [FFmpeg](https://ffmpeg.org/) for media processing.

---

*StreamSnap: Your all-in-one tool for smart media processing. Download, transcribe, and analyze with ease!*