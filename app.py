from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deepfake_cnn.keras')
model = None

def load_model():
    global model
    if model is None:
        print(f"Loading deepfake detection model from {MODEL_PATH}...")
        
        if os.path.isdir(MODEL_PATH):
            import json
            config_path = os.path.join(MODEL_PATH, 'config.json')
            weights_path = os.path.join(MODEL_PATH, 'model.weights.h5')
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model = tf.keras.models.model_from_json(json.dumps(config))
            model.load_weights(weights_path)
            print("Model loaded from directory format!")
        else:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded from file format!")
    return model

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


def analyze_deepfake(image_filepath, media_type):
    """
    Analyze media for deepfake detection using the trained CNN model.
    """
    if media_type == 'image':
        try:
            if not os.path.isfile(image_filepath):
                return {
                    'is_fake': None,
                    'confidence': 0.0,
                    'message': f'File not found or is not a file: {image_filepath}'
                }
            
            print(f"Analyzing image: {image_filepath}")
            deepfake_model = load_model()
            img_array = preprocess_image(image_filepath)
            predictions = deepfake_model.predict(img_array, verbose=0)
            
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
    
    elif media_type == 'video':
        return {
            'is_fake': None,
            'confidence': 0.0,
            'message': 'Video analysis not yet implemented. Please upload individual frames as images.'
        }
    
    elif media_type == 'audio':
        return {
            'is_fake': None,
            'confidence': 0.0,
            'message': 'Audio deepfake detection requires a separate model. Current model is for images only.'
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
