from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import librosa
import cv2

app = Flask(__name__)

IMAGE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deepfake_cnn.keras')
AUDIO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Deepfake-audio-detection')

image_model = None
audio_model = None
audio_feature_extractor = None


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


def preprocess_frame(frame):
    """Preprocess a video frame (numpy array from OpenCV) for the CNN model."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (256, 256))
    frame_array = np.array(frame_resized, dtype=np.float32) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array


def analyze_video_frames(filepath, max_frames=30):
    """
    Extract frames from video and analyze each for deepfakes.
    Returns aggregated results from all analyzed frames.
    """
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        return None, "Could not open video file"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    frame_interval = max(1, total_frames // max_frames)
    
    model = load_image_model()
    
    fake_scores = []
    real_scores = []
    frames_analyzed = 0
    
    print(f"Analyzing video: {filepath}")
    print(f"Duration: {duration:.1f}s, FPS: {fps:.1f}, Total frames: {total_frames}")
    print(f"Sampling every {frame_interval} frames (analyzing up to {max_frames} frames)")
    
    for frame_idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_array = preprocess_frame(frame)
        predictions = model.predict(frame_array, verbose=0)
        
        fake_scores.append(float(predictions[0][0]))
        real_scores.append(float(predictions[0][1]))
        frames_analyzed += 1
        
        if frames_analyzed >= max_frames:
            break
    
    cap.release()
    
    if frames_analyzed == 0:
        return None, "Could not extract any frames from video"
    
    avg_fake = np.mean(fake_scores)
    avg_real = np.mean(real_scores)
    
    return {
        'avg_fake': avg_fake,
        'avg_real': avg_real,
        'frames_analyzed': frames_analyzed,
        'duration': duration
    }, None


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
            result, error = analyze_video_frames(filepath)
            
            if error:
                return {
                    'is_fake': None,
                    'confidence': 0.0,
                    'message': f'Error analyzing video: {error}'
                }
            
            avg_fake = result['avg_fake']
            avg_real = result['avg_real']
            frames_analyzed = result['frames_analyzed']
            duration = result['duration']
            
            is_fake = bool(avg_fake > avg_real)
            confidence = avg_fake if is_fake else avg_real
            
            return {
                'is_fake': is_fake,
                'confidence': round(confidence * 100, 2),
                'fake_probability': round(avg_fake * 100, 2),
                'real_probability': round(avg_real * 100, 2),
                'frames_analyzed': frames_analyzed,
                'video_duration': round(duration, 1),
                'message': f"{'FAKE video detected' if is_fake else 'Appears REAL video'} with {confidence * 100:.1f}% confidence (analyzed {frames_analyzed} frames from {duration:.1f}s video)"
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
    app.run(debug=True, port=5000)
