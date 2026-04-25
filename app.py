from flask import Flask, render_template, request, jsonify
import os
import json
import sys
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import librosa
import cv2
from huggingface_hub import hf_hub_download

app = Flask(__name__)

IMAGE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deepfake_cnn.keras')
AUDIO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Deepfake-audio-detection')
VIDEO_MODEL_REPO_ID = os.environ.get("VIDEO_MODEL_REPO_ID", "Naman712/Deep-fake-detection")
VIDEO_MODEL_LOCAL_PATH = os.environ.get("VIDEO_MODEL_PATH", "")
VIDEO_MODEL_WEIGHTS = "model_87_acc_20_frames_final_data.pt"
VIDEO_SEQUENCE_LENGTH = 20
VIDEO_MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

image_model = None
audio_model = None
audio_feature_extractor = None
video_model = None
video_processor = None


def load_image_model():
    global image_model
    if image_model is None:
        print(f"Loading image deepfake model from {IMAGE_MODEL_PATH}...")
        
        if os.path.isdir(IMAGE_MODEL_PATH):
            config_path = os.path.join(IMAGE_MODEL_PATH, 'config.json')
            weights_path = os.path.join(IMAGE_MODEL_PATH, 'model.weights.h5')
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            image_model = tf.keras.models.model_from_json(json.dumps(config))
            image_model.load_weights(weights_path)
            print("Image model loaded!")
        else:
            image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
            print("Image model loaded!")
    return image_model


def load_audio_model():
    global audio_model, audio_feature_extractor
    if audio_model is None:
        print(f"Loading audio deepfake model from {AUDIO_MODEL_PATH}...")
        audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL_PATH)
        audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_PATH)
        audio_model.eval()
        print("Audio model loaded!")
    return audio_model, audio_feature_extractor


def load_video_model():
    global video_model, video_processor
    if video_model is not None and video_processor is not None:
        return video_model, video_processor

    print("Loading video model with strict repo parity...")

    local_model_dir = VIDEO_MODEL_LOCAL_PATH if os.path.isdir(VIDEO_MODEL_LOCAL_PATH) else None
    if local_model_dir is None:
        os.makedirs(VIDEO_MODEL_CACHE_DIR, exist_ok=True)
        token = os.environ.get("HUGGINGFACE_TOKEN")
        files_to_download = [
            "modeling_deepfake.py",
            "processor_deepfake.py",
            "modeling.py",
            VIDEO_MODEL_WEIGHTS,
            "config.json",
        ]
        for filename in files_to_download:
            local_path = os.path.join(VIDEO_MODEL_CACHE_DIR, filename)
            if not os.path.exists(local_path):
                hf_hub_download(
                    repo_id=VIDEO_MODEL_REPO_ID,
                    filename=filename,
                    token=token,
                    local_dir=VIDEO_MODEL_CACHE_DIR,
                    local_dir_use_symlinks=False,
                )
        local_model_dir = VIDEO_MODEL_CACHE_DIR

    if local_model_dir not in sys.path:
        sys.path.insert(0, local_model_dir)

    model_path = os.path.join(local_model_dir, VIDEO_MODEL_WEIGHTS)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Video model weights not found: {model_path}")

    # Model loading parity: try original HF custom class first, fallback to known architecture.
    try:
        from modeling_deepfake import DeepFakeDetectorModel
        video_model = DeepFakeDetectorModel.from_pretrained(model_path)
    except Exception:
        from modeling import DeepFakeDetector
        video_model = DeepFakeDetector(2)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        video_model.load_state_dict(state_dict)
    video_model.eval()

    # Processor loading parity: try original processor, fallback to simplified processor.
    try:
        from processor_deepfake import DeepFakeProcessor
        video_processor = DeepFakeProcessor()
    except Exception:
        class SimpleDeepFakeProcessor:
            def __init__(self, im_size=112, mean=None, std=None):
                self.im_size = im_size
                self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
                self.std = std if std is not None else [0.229, 0.224, 0.225]

            def preprocess_frame(self, frame):
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)
                frame = frame.resize((self.im_size, self.im_size))
                frame = np.array(frame).astype(np.float32) / 255.0
                frame = (frame - np.array(self.mean)) / np.array(self.std)
                frame = frame.transpose(2, 0, 1)
                return torch.tensor(frame, dtype=torch.float32)

        video_processor = SimpleDeepFakeProcessor()

    print("Video model loaded!")
    return video_model, video_processor


def extract_frames_from_video(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

# Enable CORS for Chrome extension
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def preprocess_image(filepath):
    """Preprocess image for the CNN model (256x256 RGB)."""
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_audio(filepath):
    """Load and preprocess audio for Wav2Vec2 model (16kHz)."""
    audio, sr = librosa.load(filepath, sr=16000)
    if len(audio) > 16000 * 30:
        audio = audio[:16000 * 30]
    return audio


def analyze_video_with_model(filepath):
    """
    Analyze video with strict parity to reference repo flow.
    """
    model, processor = load_video_model()
    frames = extract_frames_from_video(filepath, max_frames=VIDEO_SEQUENCE_LENGTH)
    if not frames:
        raise ValueError("Could not extract frames from video")

    processed_frames = [processor.preprocess_frame(frame) for frame in frames]

    # strict parity: enforce exactly 20 frames
    if len(processed_frames) >= VIDEO_SEQUENCE_LENGTH:
        indices = np.linspace(0, len(processed_frames) - 1, VIDEO_SEQUENCE_LENGTH, dtype=int)
        batch_frames = [processed_frames[i] for i in indices]
    else:
        batch_frames = processed_frames.copy()
        while len(batch_frames) < VIDEO_SEQUENCE_LENGTH:
            batch_frames.append(batch_frames[-1] if batch_frames else torch.zeros((3, 112, 112)))
        batch_frames = batch_frames[:VIDEO_SEQUENCE_LENGTH]

    input_tensor = torch.stack(batch_frames).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            _, logits = outputs
        else:
            logits = outputs
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # strict parity with repo backend app.py mapping:
    # index 0 = real, index 1 = fake
    real_prob = float(probs[0])
    fake_prob = float(probs[1])

    return {
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "frames_used": len(batch_frames),
    }


def analyze_deepfake(filepath, media_type):
    """
    Analyze media for deepfake detection using trained models.
    - Images: CNN model
    - Audio: Wav2Vec2 model
    """
    if not os.path.isfile(filepath):
        return {
            'is_fake': None,
            'confidence': 0.0,
            'message': f'File not found: {filepath}'
        }

    if media_type == 'image':
        try:
            print(f"Analyzing image: {filepath}")
            model = load_image_model()
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array, verbose=0)
            
            fake_prob = float(predictions[0][0])
            real_prob = float(predictions[0][1])
            
            is_fake = fake_prob > real_prob
            confidence = fake_prob if is_fake else real_prob
            
            return {
                'is_fake': is_fake,
                'confidence': round(confidence * 100, 2),
                'fake_probability': round(fake_prob * 100, 2),
                'real_probability': round(real_prob * 100, 2),
                'message': f"{'FAKE detected' if is_fake else 'Appears REAL'} with {confidence * 100:.1f}% confidence"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'is_fake': None,
                'confidence': 0.0,
                'message': f'Error analyzing image: {str(e)}'
            }
    
    elif media_type == 'audio':
        try:
            print(f"Analyzing audio: {filepath}")
            model, feature_extractor = load_audio_model()
            
            audio = preprocess_audio(filepath)
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            fake_prob = float(probs[0][0])
            real_prob = float(probs[0][1])
            
            is_fake = fake_prob > real_prob
            confidence = fake_prob if is_fake else real_prob
            
            return {
                'is_fake': is_fake,
                'confidence': round(confidence * 100, 2),
                'fake_probability': round(fake_prob * 100, 2),
                'real_probability': round(real_prob * 100, 2),
                'message': f"{'FAKE audio detected' if is_fake else 'Appears REAL audio'} with {confidence * 100:.1f}% confidence"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'is_fake': None,
                'confidence': 0.0,
                'message': f'Error analyzing audio: {str(e)}'
            }
    
    elif media_type == 'video':
        try:
            print(f"Analyzing video with dedicated model: {filepath}")
            result = analyze_video_with_model(filepath)
            fake_prob = result["fake_probability"]
            real_prob = result["real_probability"]
            is_fake = bool(fake_prob > real_prob)
            confidence = fake_prob if is_fake else real_prob
            
            return {
                'is_fake': is_fake,
                'confidence': round(confidence * 100, 2),
                'fake_probability': round(fake_prob * 100, 2),
                'real_probability': round(real_prob * 100, 2),
                'model': VIDEO_MODEL_LOCAL_PATH,
                'message': f"{'FAKE video detected' if is_fake else 'Appears REAL video'} with {confidence * 100:.1f}% confidence"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'is_fake': None,
                'confidence': 0.0,
                'message': f'Error analyzing video: {str(e)}'
            }
    
    return {
        'is_fake': None,
        'confidence': 0.0,
        'message': 'Unsupported media type'
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    media_type = request.form.get('type', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    if media_type == 'image' and not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'error': 'Invalid image format'}), 400
    elif media_type == 'video' and not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'Invalid video format'}), 400
    elif media_type == 'audio' and not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({'error': 'Invalid audio format'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    print(f"File saved to: {upload_path}")
    
    # Analyze with ML model
    result = analyze_deepfake(upload_path, media_type)
    
    # Clean up uploaded file (optional)
    # os.remove(filepath)
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5002)
