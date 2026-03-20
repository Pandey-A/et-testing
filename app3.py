from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
import logging
from functools import lru_cache

# === Configure logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Paths (update as needed) ===
VIDEO_MODEL_PATH = "video_deepfake.h5"
IMAGE_MODEL_PATH = "xception_deepfake_image.h5"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Constants ===
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# === Initialize Flask app ===
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins; restrict later in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# === Global variables for models ===
feature_extractor = None
video_model = None
image_model = None

# === Utility functions ===

def crop_center_square(frame):
    try:
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]
    except Exception as e:
        logger.error(f"Error in crop_center_square: {e}")
        raise

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    logger.info(f"Loading video from: {path}")
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

@lru_cache(maxsize=1)
def build_feature_extractor():
    logger.info("Building feature extractor...")
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    model = keras.Model(inputs, outputs, name="feature_extractor")
    logger.info("Feature extractor built")
    return model

def build_video_model(max_seq_length, num_features):
    logger.info("Building video model...")
    frame_features_input = keras.Input((max_seq_length, num_features))
    mask_input = keras.Input((max_seq_length,), dtype="bool")
    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model([frame_features_input, mask_input], output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    logger.info("Video model built")
    return model

def prepare_single_video(frames):
    logger.info(f"Preparing video with {len(frames)} frames")
    if len(frames) > MAX_SEQ_LENGTH:
        indices = np.linspace(0, len(frames) - 1, MAX_SEQ_LENGTH, dtype=int)
        frames = frames[indices]
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        batch_size = 5
        for start_idx in range(0, length, batch_size):
            end_idx = min(start_idx + batch_size, length)
            for j in range(start_idx, end_idx):
                frame_batch = np.expand_dims(batch[j], axis=0)
                features = feature_extractor.predict(frame_batch, verbose=0)
                frame_features[i, j, :] = features[0]
        frame_mask[i, :length] = 1
    return frame_features, frame_mask

def sequence_prediction(video_path):
    logger.info(f"Starting video prediction for {video_path}")
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames)
    pred = video_model.predict([frame_features, frame_mask], verbose=0)
    return float(pred[0][0])

def initialize_models():
    global feature_extractor, video_model, image_model
    logger.info("Initializing models...")
    if not os.path.exists(VIDEO_MODEL_PATH):
        logger.error(f"Video model file not found: {VIDEO_MODEL_PATH}")
        raise FileNotFoundError(f"{VIDEO_MODEL_PATH} missing")
    if not os.path.exists(IMAGE_MODEL_PATH):
        logger.error(f"Image model file not found: {IMAGE_MODEL_PATH}")
        raise FileNotFoundError(f"{IMAGE_MODEL_PATH} missing")
    feature_extractor = build_feature_extractor()
    video_model = build_video_model(MAX_SEQ_LENGTH, NUM_FEATURES)
    video_model.load_weights(VIDEO_MODEL_PATH)
    image_model = keras.models.load_model(IMAGE_MODEL_PATH)
    logger.info("All models loaded successfully")

# === Image prediction logic ===

def predict_fake_or_real(img_path):
    logger.info(f"Starting image prediction for {img_path}")
    img = keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    pred = image_model.predict(img_array, verbose=0)[0][0]
    label = "FAKE" if pred >= 0.5 else "REAL"
    logger.info(f"Image prediction done. Score: {pred}, Label: {label}")
    return label, float(pred)

# === Flask routes ===

@app.route('/')
def home():
    return jsonify({"status": "running", "models_loaded": feature_extractor is not None and video_model is not None and image_model is not None})

@app.route('/health', methods=['GET'])
def health_check():
    try:
        models_ready = feature_extractor is not None and video_model is not None and image_model is not None
        return jsonify({
            "status": "healthy" if models_ready else "models_not_loaded",
            "models_loaded": models_ready,
            "upload_folder_exists": os.path.exists(UPLOAD_FOLDER)
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/predict/video', methods=['POST'])
def predict_video():
    try:
        if feature_extractor is None or video_model is None:
            initialize_models()
        
        if 'video' not in request.files:
            return jsonify({"error": "No video uploaded"}), 400
        video = request.files['video']
        if video.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        allowed = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if not any(video.filename.lower().endswith(ext) for ext in allowed):
            return jsonify({"error": "Unsupported video format"}), 400
        
        filename = secure_filename(video.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(file_path)
        
        score = sequence_prediction(file_path)
        label = "FAKE" if score >= 0.5 else "REAL"
        confidence = score if score >= 0.5 else 1 - score
        
        os.remove(file_path)
        
        return jsonify({
            "filename": filename,
            "score": score,
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        logger.error(f"Error in video prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/image', methods=['POST'])
def predict_image():
    try:
        if image_model is None:
            initialize_models()
        
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        allowed = {'.jpg', '.jpeg', '.png', '.bmp'}
        if not any(image_file.filename.lower().endswith(ext) for ext in allowed):
            return jsonify({"error": "Unsupported image format"}), 400
        
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        
        label, score = predict_fake_or_real(file_path)
        
        os.remove(file_path)
        
        return jsonify({
            "filename": filename,
            "score": score,
            "prediction": label
        })
    except Exception as e:
        logger.error(f"Error in image prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max size is 100MB."}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({"error": "Internal server error"}), 500

# === Run application ===

try:
    initialize_models()
except Exception as e:
    logger.error(f"Model initialization failed: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
