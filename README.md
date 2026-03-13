# Deepfake Detection UI

A simple web interface and Chrome extension for detecting deepfakes in images, videos, and audio files.

## Setup

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Run the application:
```bash
python3 app.py
```

3. Open your browser and go to: `http://localhost:5000`

## Chrome Extension

### Install the Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top right)
3. Click **Load unpacked**
4. Select the `extension` folder from this project

### Using the Extension

1. Make sure the Flask server is running (`python3 app.py`)
2. Click the extension icon in Chrome toolbar
3. Upload an image, video, or audio file
4. Click "Analyze" to detect deepfakes

## Features

- **Image Analysis**: PNG, JPG, JPEG, GIF, WebP
- **Video Analysis**: MP4, AVI, MOV, MKV, WebM
- **Audio Analysis**: MP3, WAV, OGG, FLAC, M4A

## Integrating Your ML Model

Edit the `analyze_deepfake()` function in `app.py`:

```python
def analyze_deepfake(filepath, media_type):
    if media_type == 'image':
        result = your_image_model.predict(filepath)
    elif media_type == 'video':
        result = your_video_model.predict(filepath)
    elif media_type == 'audio':
        result = your_audio_model.predict(filepath)
    
    return {
        'is_fake': True/False,
        'confidence': 0.0-1.0,
        'message': 'Optional message'
    }
```

## Project Structure

```
ml_Cp/
├── app.py              # Flask backend
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web UI
├── extension/          # Chrome extension
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   ├── styles.css
│   └── icon*.png
├── uploads/            # Uploaded files (auto-created)
└── README.md
```
