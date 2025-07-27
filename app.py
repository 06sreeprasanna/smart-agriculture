from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
from datetime import datetime
import cv2

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load models and related files
def load_models():
    """Load all required models and supporting files"""
    try:
        # Soil model components
        with open('models/soil_model.pkl', 'rb') as f:
            soil_model = pickle.load(f)
        
        with open('models/soil_scaler.pkl', 'rb') as f:
            soil_scaler = pickle.load(f)
        
        with open('models/soil_types.pkl', 'rb') as f:
            soil_types = pickle.load(f)
        
        # Plant disease model components
        disease_model = load_model('models/disease_model_best.h5')
        
        with open('models/disease_classes.pkl', 'rb') as f:
            disease_classes = pickle.load(f)
        
        logger.info("All models loaded successfully")
        return soil_model, soil_scaler, soil_types, disease_model, disease_classes
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Load models at startup
try:
    soil_model, soil_scaler, soil_types, disease_model, disease_classes = load_models()
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise SystemExit("Could not start application due to model loading failure")


# Helper functions
def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image for disease prediction"""
    try:
        # Read and resize image
        img = cv2.imread(image_path)
        img = cv2.resize(img, target_size)
        
        # Convert to RGB (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def get_soil_recommendations(soil_type):
    """Get crop recommendations based on soil type"""
    recommendations = {
        'Sandy': [
            "Carrots, Radishes, Potatoes (root crops that can penetrate loose soil)",
            "Asparagus (tolerates sandy conditions)",
            "Tomatoes (with proper irrigation)",
            "Melons and Squash (warm-season crops)",
            "Add organic matter to improve water retention"
        ],
        'Loamy': [
            "Most vegetables: Tomatoes, Peppers, Cucumbers",
            "Leafy greens: Lettuce, Spinach",
            "Root crops: Carrots, Beets",
            "Legumes: Beans, Peas",
            "This is ideal soil - most crops will thrive"
        ],
        'Clay': [
            "Brassicas: Cabbage, Broccoli, Brussels sprouts",
            "Fruit trees (they can anchor well in clay)",
            "Perennials with strong root systems",
            "Add organic matter and sand to improve drainage",
            "Avoid root crops that need loose soil"
        ],
        'Silty': [
            "Most crops do well in silty soil",
            "Leafy greens: Lettuce, Kale",
            "Root crops: Onions, Garlic",
            "Corn and other tall crops",
            "Be careful with compaction - avoid working soil when wet"
        ],
        'Peaty': [
            "Acid-loving plants: Blueberries, Cranberries",
            "Potatoes (tolerate acidic soil)",
            "Add lime to raise pH for most vegetables",
            "Improve drainage with sand or grit",
            "Avoid crops that need alkaline conditions"
        ],
        'Chalky': [
            "Alkaline-loving plants: Lavender, Lilac",
            "Brassicas: Cabbage, Kale",
            "Spinach and Beets",
            "Add organic matter to improve water retention",
            "Avoid acid-loving plants"
        ]
    }
    return recommendations.get(soil_type, ["No specific recommendations available"])

def get_disease_remedies(disease_class):
    """Get treatment recommendations for plant diseases"""
    remedies = {
        'healthy': "Plant is healthy. Continue current care practices.",
        'powdery_mildew': [
            "Apply fungicides containing sulfur, potassium bicarbonate, or neem oil",
            "Improve air circulation around plants",
            "Remove and destroy infected plant parts",
            "Avoid overhead watering",
            "Plant resistant varieties in future"
        ],
        'late_blight': [
            "Apply copper-based fungicides",
            "Remove and destroy infected plants immediately",
            "Avoid overhead watering",
            "Ensure proper spacing for air circulation",
            "Rotate crops annually"
        ],
        'leaf_spot': [
            "Apply fungicides containing chlorothalonil or copper",
            "Remove infected leaves and debris",
            "Water at the base of plants",
            "Improve air circulation",
            "Avoid working with wet plants"
        ],
        'bacterial_spot': [
            "Apply copper-based bactericides",
            "Remove and destroy severely infected plants",
            "Avoid overhead irrigation",
            "Rotate with non-host crops",
            "Use disease-free seeds"
        ],
        'rust': [
            "Apply fungicides containing myclobutanil or tebuconazole",
            "Remove infected leaves promptly",
            "Space plants for good air flow",
            "Water early in the day",
            "Plant resistant varieties"
        ]
    }
    
    # Check if exact match exists
    if disease_class in remedies:
        return remedies[disease_class]
    
    # Check for partial matches (common for plant disease names)
    for key, value in remedies.items():
        if key in disease_class.lower():
            return value
    
    # Default advice if no specific remedy found
    return [
        "Remove and destroy infected plant parts",
        "Apply broad-spectrum fungicide",
        "Improve air circulation around plants",
        "Avoid overhead watering",
        "Consider planting resistant varieties next season"
    ]

# Routes
@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/soil-analysis', methods=['GET', 'POST'])
def soil_analysis():
    """Handle soil analysis form"""
    if request.method == 'POST':
        try:
            # Get form data
            n = float(request.form['nitrogen'])
            p = float(request.form['phosphorus'])
            k = float(request.form['potassium'])
            ph = float(request.form['ph'])
            
            # Prepare input for model
            input_data = np.array([[n, p, k, ph]])
            scaled_data = soil_scaler.transform(input_data)
            
            # Make prediction
            prediction = soil_model.predict(scaled_data)
            soil_type_code = prediction[0]
            soil_type = soil_types.get(soil_type_code, 'Unknown')
            
            # Get recommendations
            recommendations = get_soil_recommendations(soil_type)
            
            # Prepare results
            result = {
                'soil_type': soil_type,
                'input_values': {
                    'Nitrogen (N)': f"{n} ppm",
                    'Phosphorus (P)': f"{p} ppm",
                    'Potassium (K)': f"{k} ppm",
                    'pH': ph
                },
                'recommendations': recommendations
            }
            
            logger.info(f"Soil analysis result: {result}")
            return render_template('soil_result.html', result=result)
        
        except Exception as e:
            logger.error(f"Soil analysis error: {str(e)}")
            return render_template('error.html', message="Invalid input data. Please check your values and try again.")
    
    return render_template('soil_form.html')

@app.route('/disease-detection', methods=['POST'])
def disease_detection():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', filename)
        file.save(temp_path)

        try:
            # Get prediction from your model
            disease, confidence = predict_disease(temp_path)
            
            # Ensure confidence is float
            confidence = float(confidence)
            
            # Debug output
            print(f"Prediction Result - Disease: {disease}, Confidence: {confidence}")
            
            return render_template('disease.html',
                                disease=disease,
                                confidence=confidence,
                                image_filename=filename)  # Pass filename for display
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return render_template('disease.html',
                                error=str(e))
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
@app.route('/api/soil', methods=['POST'])
def api_soil_analysis():
    """API endpoint for soil analysis"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'ph']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Prepare input
        input_data = np.array([[data['nitrogen'], data['phosphorus'], data['potassium'], data['ph']]])
        scaled_data = soil_scaler.transform(input_data)
        
        # Make prediction
        prediction = soil_model.predict(scaled_data)
        soil_type_code = prediction[0]
        soil_type = soil_types.get(soil_type_code, 'Unknown')
        
        return jsonify({
            'soil_type': soil_type,
            'soil_type_code': int(soil_type_code),
            'input_values': {
                'nitrogen': data['nitrogen'],
                'phosphorus': data['phosphorus'],
                'potassium': data['potassium'],
                'ph': data['ph']
            }
        })
    
    except Exception as e:
        logger.error(f"API soil analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/disease', methods=['POST'])
def api_disease_detection():
    """API endpoint for disease detection"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess image
            processed_img = preprocess_image(filepath)
            
            # Make prediction
            predictions = disease_model.predict(processed_img)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get class name
            predicted_class = disease_classes[predicted_class_idx]
            
            # Clean up - remove temporary file
            os.remove(filepath)
            
            return jsonify({
                'disease': predicted_class,
                'confidence': confidence,
                'is_healthy': 'healthy' in predicted_class.lower()
            })
        
        return jsonify({'error': 'File type not allowed'}), 400
    
    except Exception as e:
        logger.error(f"API disease detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)