# Deepfake Detection App

A web application for detecting deepfakes in images, videos, and audio using CNN and Wav2Vec2 models.

## Models

| Media Type | Model | Architecture |
|------------|-------|--------------|
| **Image** | `deepfake_cnn.keras` (included) | CNN with 4 Conv2D blocks |
| **Video** | Uses image model | Frame-by-frame analysis |
| **Audio** | [Wav2Vec2](https://huggingface.co/mo-thecreator/Deepfake-audio-detection) | Transformer-based |

## Setup

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Download the audio model (optional, for audio detection):
```bash
git lfs install
git clone https://huggingface.co/mo-thecreator/Deepfake-audio-detection
```

3. Run the application:
```bash
python3 app.py
```

4. Open your browser: `http://localhost:5000`

## Features

- **Image Analysis**: Analyzes images for deepfake artifacts using CNN
- **Video Analysis**: Extracts frames and analyzes each with the image model
- **Audio Analysis**: Uses Wav2Vec2 transformer model for voice deepfake detection

### Supported Formats

- **Images**: PNG, JPG, JPEG, GIF, BMP, WebP
- **Videos**: MP4, AVI, MOV, MKV, WebM
- **Audio**: MP3, WAV, OGG, FLAC, M4A

## Chrome Extension

1. Open Chrome → `chrome://extensions/`
2. Enable **Developer mode**
3. Click **Load unpacked** → Select the `extension` folder
4. Make sure Flask server is running, then use the extension

## API

**POST /analyze**

```bash
curl -X POST -F "file=@image.jpg" -F "type=image" http://localhost:5000/analyze
```

Response:
```json
{
  "is_fake": true,
  "confidence": 87.5,
  "fake_probability": 87.5,
  "real_probability": 12.5,
  "message": "FAKE detected with 87.5% confidence"
}
```

## Project Structure

```
ml_Cp/
├── app.py                    # Flask backend with ML inference
├── requirements.txt          # Python dependencies
├── deepfake_cnn.keras/       # Image deepfake CNN model (included)
├── Deepfake-audio-detection/ # Audio model (download separately)
├── templates/
│   └── index.html            # Web UI
├── extension/                # Chrome extension
└── uploads/                  # Uploaded files (auto-created)
```

## Requirements

- Python 3.9+
- TensorFlow 2.16+
- PyTorch 2.1+
- ~1GB disk space for models
