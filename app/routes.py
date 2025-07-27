from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
import pickle
import cv2
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from .models import YieldData
from .utils.weather_api import get_weather_data
from .utils.market_api import get_market_prices
from .utils.notification import send_sms_alert
from . import db
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint instance
bp = Blueprint('main', __name__)

# Configuration
MODEL_DIR = 'models'
SOIL_MODEL_PATH = os.path.join(MODEL_DIR, 'soil_model.pkl')
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, 'disease_model.h5')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize models
soil_model = None
disease_model = None
models_loaded = False

# Default class mappings
soil_types = {
    0: 'Sandy',
    1: 'Loamy',
    2: 'Clay',
    3: 'Silty',
    4: 'Peaty',
    5: 'Chalky'
}

disease_classes = {
    0: 'Apple Scab',
    1: 'Apple Black Rot',
    2: 'Apple Cedar Rust',
    3: 'Apple Healthy',
    4: 'Blueberry Healthy',
    5: 'Cherry Powdery Mildew',
    6: 'Corn Gray Leaf Spot',
    7: 'Peach Bacterial Spot',
    8: 'Potato Early Blight',
    9: 'Tomato Late Blight'
}

def handle_model_errors(f):
    """Decorator to handle model-related errors gracefully"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            flash('Service temporarily unavailable. Please try again later.', 'danger')
            return redirect(url_for('main.index'))
    return wrapper

def create_emergency_soil_model():
    """Create minimal viable soil model with improved dummy data"""
    global soil_model
    try:
        soil_model = RandomForestClassifier(n_estimators=100, random_state=42)
        # More comprehensive dummy data
        X_dummy = [
            [25,15,120,6.5], [30,20,150,6.8], [35,25,180,7.2],
            [28,18,140,6.9], [22,12,110,6.3], [20,10,100,7.5],
            [18,8,90,5.8], [32,22,160,7.0], [27,17,130,6.7],
            [23,13,115,6.4], [19,9,95,5.9], [31,21,155,6.9]
        ]
        y_dummy = [0,1,2,3,4,5,0,1,2,3,4,5]
        soil_model.fit(X_dummy, y_dummy)
        logger.warning("Emergency soil model created with extended dataset")
    except Exception as e:
        logger.error(f"Failed to create emergency soil model: {str(e)}")
        raise

def create_emergency_disease_model():
    """Create enhanced emergency disease model"""
    global disease_model
    try:
        disease_model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(256, 256, 3)),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(disease_classes), activation='softmax')
        ])
        disease_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.warning("Enhanced emergency disease model created")
    except Exception as e:
        logger.error(f"Failed to create emergency disease model: {str(e)}")
        raise

def load_models():
    """Load models with comprehensive fallbacks and periodic refresh"""
    global soil_model, disease_model, models_loaded
    
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Load soil model with version checking
        try:
            if os.path.exists(SOIL_MODEL_PATH):
                with open(SOIL_MODEL_PATH, 'rb') as f:
                    soil_model = pickle.load(f)
                logger.info(f"Soil model loaded successfully (Version: {soil_model.__dict__.get('version', 'unknown')})")
            else:
                logger.warning("Soil model file not found")
                create_emergency_soil_model()
        except Exception as e:
            logger.error(f"Error loading soil model: {str(e)}")
            create_emergency_soil_model()
        
        # Load disease model with validation
        try:
            if os.path.exists(DISEASE_MODEL_PATH):
                disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
                try:
                    # Simple validation check
                    dummy_input = np.random.rand(1, 256, 256, 3)
                    disease_model.predict(dummy_input)
                    logger.info("Disease model loaded and validated successfully")
                except Exception as e:
                    logger.warning("Disease model failed validation, creating emergency model")
                    create_emergency_disease_model()
            else:
                logger.warning("Disease model file not found")
                create_emergency_disease_model()
        except Exception as e:
            logger.error(f"Error loading disease model: {str(e)}")
            create_emergency_disease_model()
        
        models_loaded = True
        
    except Exception as e:
        logger.critical(f"Critical model loading error: {str(e)}")
        try:
            create_emergency_soil_model()
            create_emergency_disease_model()
        except Exception as e:
            logger.critical(f"Failed to create emergency models: {str(e)}")
        models_loaded = bool(soil_model and disease_model)

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@bp.route('/check-model')
def check_model():
    """Endpoint to check model status for debugging"""
    return jsonify({
        'disease_model_loaded': bool(disease_model),
        'soil_model_loaded': bool(soil_model),
        'disease_model_path': DISEASE_MODEL_PATH,
        'soil_model_path': SOIL_MODEL_PATH,
        'disease_model_exists': os.path.exists(DISEASE_MODEL_PATH),
        'soil_model_exists': os.path.exists(SOIL_MODEL_PATH),
        'models_loaded': models_loaded,
        'status': 'success'
    })

@bp.route('/soil-detection', methods=['GET', 'POST'])
@handle_model_errors
def soil_detection():
    """Soil analysis route with enhanced validation"""
    if request.method == 'POST':
        if not soil_model:
            create_emergency_soil_model()
            if not soil_model:
                flash('Soil analysis unavailable. Please try later.', 'danger')
                return redirect(url_for('main.index'))
            flash('Using limited functionality mode', 'warning')

        try:
            # Validate and parse input
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])
            ph = float(request.form['ph'])
            
            # Enhanced input validation
            if not (0 <= nitrogen <= 100):
                raise ValueError("Nitrogen must be between 0-100")
            if not (0 <= phosphorus <= 100):
                raise ValueError("Phosphorus must be between 0-100")
            if not (0 <= potassium <= 200):
                raise ValueError("Potassium must be between 0-200")
            if not (0 <= ph <= 14):
                raise ValueError("pH must be between 0-14")
                
        except ValueError as e:
            flash(f'Invalid input: {str(e)}', 'danger')
            return redirect(url_for('main.soil_detection'))

        try:
            input_data = np.array([[nitrogen, phosphorus, potassium, ph]])
            prediction = soil_model.predict(input_data)[0]
            soil_type = soil_types.get(prediction, 'Unknown')
            
            # Get probabilities for all classes
            if hasattr(soil_model, 'predict_proba'):
                probabilities = soil_model.predict_proba(input_data)[0]
                confidence = {soil_types[i]: f"{prob*100:.1f}%" 
                             for i, prob in enumerate(probabilities)}
            else:
                confidence = None
            
            return render_template('soil.html',
                                soil_type=soil_type,
                                confidence=confidence,
                                crops=suggest_crops(soil_type),
                                show_result=True)
            
        except Exception as e:
            logger.error(f"Soil detection error: {str(e)}", exc_info=True)
            flash('Analysis failed. Please try again.', 'danger')
    
    return render_template('soil.html', show_result=False)

def suggest_crops(soil_type):
    """Get crop suggestions based on soil type with more detailed recommendations"""
    suggestions = {
        'Sandy': {
            'best': 'Carrots, Radishes, Potatoes, Asparagus',
            'tips': 'Sandy soil drains quickly. Add organic matter to improve water retention.'
        },
        'Loamy': {
            'best': 'Wheat, Sugarcane, Cotton, Tomatoes, Corn',
            'tips': 'Loamy soil is ideal for most crops. Maintain organic content.'
        },
        'Clay': {
            'best': 'Rice, Kale, Lettuce, Broccoli, Cabbage',
            'tips': 'Clay retains water. Improve drainage with organic matter.'
        },
        'Silty': {
            'best': 'Wheat, Barley, Oats, Onions, Garlic',
            'tips': 'Silty soil is fertile but can compact easily.'
        },
        'Peaty': {
            'best': 'Root vegetables, Salad crops, Blueberries',
            'tips': 'Peaty soil is acidic. May need lime for some crops.'
        },
        'Chalky': {
            'best': 'Spinach, Beets, Sweet Corn, Lavender',
            'tips': 'Chalky soil is alkaline. Add organic matter to improve.'
        }
    }
    return suggestions.get(soil_type, {
        'best': 'Consult local agriculture expert',
        'tips': 'Soil type not recognized'
    })

@bp.route('/disease-detection', methods=['GET', 'POST'])
@handle_model_errors
def disease_detection():
    """Plant disease detection route with enhanced debugging"""
    if request.method == 'POST':
        logger.info("\n=== New Disease Detection Request ===")
        
        if not disease_model:
            logger.warning("No primary model available - creating emergency model")
            create_emergency_disease_model()
            if not disease_model:
                flash('Disease detection unavailable. Please try later.', 'danger')
                return redirect(url_for('main.index'))
            flash('Using limited functionality mode', 'warning')

        file = request.files.get('file')
        logger.info(f"File received: {file.filename if file else 'No file'}")
        
        if not file or file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('main.disease_detection'))
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Allowed: png, jpg, jpeg, gif', 'danger')
            return redirect(url_for('main.disease_detection'))

        try:
            filename = secure_filename(file.filename)
            upload_dir = current_app.config.get('UPLOAD_FOLDER', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            upload_path = os.path.join(upload_dir, filename)
            file.save(upload_path)
            logger.info(f"File saved to: {upload_path}")

            # Enhanced image processing
            img = cv2.imread(upload_path)
            logger.info(f"Image loaded: {img is not None}")
            if img is not None:
                logger.info(f"Image shape: {img.shape}")
                
            if img is None:
                raise ValueError("Invalid image file")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            
            # Image preprocessing pipeline
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) / 255.0
            logger.info(f"Model input shape: {img_array.shape}")
            
            predictions = disease_model.predict(img_array)
            logger.info(f"Raw predictions: {predictions}")
            
            pred_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0])) * 100
            disease = disease_classes.get(pred_class, 'Unknown')
            logger.info(f"Prediction: {disease} with {confidence:.2f}% confidence")
            
            # Get top 3 predictions if confidence is low
            additional_predictions = []
            if confidence < 70:
                top3 = np.argsort(predictions[0])[-3:][::-1]
                additional_predictions = [
                    (disease_classes.get(i, 'Unknown'), f"{predictions[0][i]*100:.1f}%")
                    for i in top3 if i != pred_class
                ]
                logger.info(f"Additional predictions: {additional_predictions}")
            
            return render_template('disease.html',
                                disease=disease,
                                confidence=f"{confidence:.1f}%",
                                additional_predictions=additional_predictions,
                                treatment=get_treatment(disease),
                                image_path=f"uploads/{filename}",
                                show_result=True)
            
        except Exception as e:
            logger.error(f"Disease detection error: {str(e)}", exc_info=True)
            flash('Detection failed. Please try a different image.', 'danger')
        finally:
            if 'upload_path' in locals() and os.path.exists(upload_path):
                try:
                    os.remove(upload_path)
                    logger.info("Temporary file cleaned up")
                except Exception as e:
                    logger.error(f"Error deleting temp file: {str(e)}")
    
    return render_template('disease.html', show_result=False)

def get_treatment(disease):
    """Get treatment recommendations for plant diseases with prevention tips"""
    treatments = {
        'Apple Scab': {
            'treatment': 'Apply fungicides containing myclobutanil or sulfur.',
            'prevention': 'Remove fallen leaves in autumn to reduce spores.'
        },
        'Apple Black Rot': {
            'treatment': 'Prune infected branches and apply fungicides.',
            'prevention': 'Ensure good air circulation around trees.'
        },
        'Apple Cedar Rust': {
            'treatment': 'Apply fungicides in early spring.',
            'prevention': 'Remove nearby junipers which harbor the fungus.'
        },
        'Cherry Powdery Mildew': {
            'treatment': 'Apply sulfur or potassium bicarbonate.',
            'prevention': 'Plant resistant varieties when possible.'
        },
        'Corn Gray Leaf Spot': {
            'treatment': 'Use fungicides containing azoxystrobin.',
            'prevention': 'Rotate crops and till residue after harvest.'
        },
        'Tomato Late Blight': {
            'treatment': 'Remove infected plants, use copper fungicides.',
            'prevention': 'Avoid overhead watering and space plants properly.'
        }
    }
    return treatments.get(disease, {
        'treatment': 'Consult local agriculture expert.',
        'prevention': 'Maintain good plant hygiene and monitor regularly.'
    })

@bp.route('/smart-irrigation', methods=['GET', 'POST'])
def smart_irrigation():
    """Smart irrigation recommendation system with enhanced logic"""
    if request.method == 'POST':
        try:
            location = request.form.get('location', '').strip()
            crop_type = request.form.get('crop_type', '').strip()
            growth_stage = request.form.get('growth_stage', 'mature').strip()
            
            try:
                soil_moisture = float(request.form.get('soil_moisture', 0))
                if not (0 <= soil_moisture <= 100):
                    raise ValueError("Soil moisture must be 0-100%")
            except ValueError as e:
                flash(f'Invalid soil moisture: {str(e)}', 'danger')
                return redirect(url_for('main.smart_irrigation'))
            
            if not location or not crop_type:
                flash('Please fill in all required fields', 'danger')
                return redirect(url_for('main.smart_irrigation'))
            
            # Get weather data with fallback
            try:
                weather = get_weather_data(location) or {
                    'temperature': 25,
                    'humidity': 60,
                    'precipitation': 0,
                    'condition': 'Unknown'
                }
            except Exception as e:
                logger.warning(f"Weather API error: {str(e)}")
                weather = {
                    'temperature': 25,
                    'humidity': 60,
                    'precipitation': 0,
                    'condition': 'Unknown'
                }
                flash('Weather data unavailable. Using default values.', 'warning')
            
            recommendation = calculate_irrigation(
                soil_moisture,
                weather.get('temperature', 25),
                weather.get('humidity', 60),
                weather.get('precipitation', 0),
                crop_type,
                growth_stage
            )
            
            return render_template('irrigation.html',
                                recommendation=recommendation,
                                weather=weather,
                                show_result=True)
        
        except Exception as e:
            logger.error(f"Irrigation error: {str(e)}", exc_info=True)
            flash(f'Error processing irrigation data: {str(e)}', 'danger')
    
    return render_template('irrigation.html', show_result=False)

def calculate_irrigation(moisture, temp, humidity, precip, crop_type, growth_stage='mature'):
    """Enhanced irrigation calculation with growth stage consideration"""
    water_needy = ['rice', 'sugarcane', 'banana']
    drought_tolerant = ['millet', 'sorghum', 'cactus']
    
    # Adjust moisture thresholds based on temperature
    temp_adjustment = max(0, (temp - 25) / 5)  # Increase threshold for higher temps
    moisture_threshold = 40 + temp_adjustment * 5
    
    # Growth stage multipliers
    growth_multiplier = {
        'seedling': 0.7,
        'vegetative': 0.9,
        'flowering': 1.2,
        'fruiting': 1.1,
        'mature': 1.0
    }.get(growth_stage.lower(), 1.0)
    
    if moisture < (25 * growth_multiplier):
        base_recommendation = "Immediate irrigation needed"
    elif moisture < (45 * growth_multiplier):
        base_recommendation = "Recommended to irrigate today"
    elif moisture < (moisture_threshold * growth_multiplier) and precip < 5:
        base_recommendation = "Consider irrigation in next 2 days"
    else:
        base_recommendation = "No irrigation needed currently"
    
    # Crop-specific adjustments
    if crop_type.lower() in water_needy:
        if "No irrigation" in base_recommendation:
            return "Consider light irrigation (crop is water-needy)"
        elif "Consider" in base_recommendation:
            return "Recommended irrigation (crop is water-needy)"
    elif crop_type.lower() in drought_tolerant:
        if "Immediate" in base_recommendation:
            return "Consider irrigation (crop is drought-tolerant)"
        elif "Recommended" in base_recommendation:
            return "Consider light irrigation (crop is drought-tolerant)"
    
    return base_recommendation

@bp.route('/yield-tracker', methods=['GET', 'POST'])
def yield_tracker():
    """Enhanced yield tracking system with statistics"""
    if request.method == 'POST':
        try:
            crop = request.form.get('crop', '').strip()
            
            try:
                area = float(request.form.get('area', 0))
                yield_amount = float(request.form.get('yield', 0))
                if area <= 0 or yield_amount <= 0:
                    raise ValueError("Area and yield must be positive numbers")
            except ValueError as e:
                flash(f'Invalid input: {str(e)}', 'danger')
                return redirect(url_for('main.yield_tracker'))
            
            date = request.form.get('date', '')
            if not date:
                flash('Please select a valid date', 'danger')
                return redirect(url_for('main.yield_tracker'))
            
            if not crop:
                flash('Please enter a crop name', 'danger')
                return redirect(url_for('main.yield_tracker'))
            
            new_record = YieldData(
                crop=crop,
                area=area,
                yield_amount=yield_amount,
                date=date
            )
            
            db.session.add(new_record)
            db.session.commit()
            
            # Calculate yield per unit area
            yield_per_area = yield_amount / area if area > 0 else 0
            flash(f'Yield data added successfully! Yield: {yield_per_area:.2f} units/area', 'success')
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving yield data: {str(e)}", exc_info=True)
            flash(f'Error saving yield data: {str(e)}', 'danger')
    
    # Get yield history with calculated yield/area
    history = []
    try:
        records = YieldData.query.order_by(YieldData.date.desc()).limit(20).all()
        history = [{
            'id': record.id,
            'crop': record.crop,
            'area': record.area,
            'yield_amount': record.yield_amount,
            'date': record.date,
            'yield_per_area': record.yield_amount / record.area if record.area > 0 else 0
        } for record in records]
    except Exception as e:
        logger.error(f"Error retrieving yield history: {str(e)}")
        flash('Error loading yield history', 'danger')
    
    return render_template('yield.html', history=history)

@bp.route('/market-prices')
def market_prices():
    """Market price information with enhanced error handling"""
    location = request.args.get('location', 'default').strip()
    try:
        prices = get_market_prices(location) or []
        logger.info(f"Retrieved {len(prices)} market prices for {location}")
    except Exception as e:
        logger.error(f"Error retrieving market prices: {str(e)}")
        flash('Error retrieving market prices', 'danger')
        prices = []
    return render_template('market.html', prices=prices)

@bp.route('/send-alert', methods=['POST'])
def send_alert():
    """SMS alert system with enhanced validation"""
    phone = request.form.get('phone', '').strip()
    message = request.form.get('message', '').strip()
    
    if not phone or not message:
        flash('Phone number and message are required', 'danger')
        return redirect(url_for('main.index'))
    
    if len(message) > 160:
        flash('Message too long (max 160 characters)', 'danger')
        return redirect(url_for('main.index'))
    
    try:
        send_sms_alert(phone, message)
        logger.info(f"SMS alert sent to {phone}")
        flash('Alert sent successfully!', 'success')
    except Exception as e:
        logger.error(f"Failed to send alert to {phone}: {str(e)}")
        flash(f'Failed to send alert: {str(e)}', 'danger')
    
    return redirect(url_for('main.index'))

# Initialize models when blueprint is loaded
load_models()