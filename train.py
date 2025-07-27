import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)
import pickle
import os
import cv2
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
SOIL_DATA_PATH = 'data/soil_data.csv'
PLANTVILLAGE_PATH = 'data/plantvillage'

def setup_directories():
    """Create required directories if they don't exist"""
    os.makedirs('models', exist_ok=True)
    os.makedirs(PLANTVILLAGE_PATH, exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)

def load_or_generate_soil_data():
    """Load soil data or generate synthetic data if not available"""
    try:
        if os.path.exists(SOIL_DATA_PATH):
            df = pd.read_csv(SOIL_DATA_PATH)
            logger.info(f"Loaded soil data from {SOIL_DATA_PATH}")
            return df
        
        logger.info(f"{SOIL_DATA_PATH} not found. Generating synthetic data...")
        
        # Enhanced synthetic data with realistic agricultural ranges
        np.random.seed(SEED)
        num_samples = 5000
        
        soil_ranges = {
            'Sandy': {'N': (15, 40), 'P': (10, 30), 'K': (80, 150), 'pH': (5.5, 7.0)},
            'Loamy': {'N': (25, 60), 'P': (15, 40), 'K': (120, 200), 'pH': (6.0, 7.5)},
            'Clay': {'N': (30, 80), 'P': (20, 50), 'K': (150, 300), 'pH': (6.5, 8.0)},
            'Silty': {'N': (20, 50), 'P': (15, 35), 'K': (100, 180), 'pH': (6.0, 7.5)},
            'Peaty': {'N': (10, 30), 'P': (5, 20), 'K': (50, 120), 'pH': (4.5, 6.5)},
            'Chalky': {'N': (20, 45), 'P': (10, 25), 'K': (100, 160), 'pH': (7.5, 9.0)}
        }
        
        data = []
        for soil_type, (soil_name) in enumerate(soil_ranges.keys()):
            ranges = soil_ranges[soil_name]
            n_samples = num_samples // len(soil_ranges)
            
            for _ in range(n_samples):
                data.append([
                    np.random.uniform(*ranges['N']),
                    np.random.uniform(*ranges['P']),
                    np.random.uniform(*ranges['K']),
                    np.random.uniform(*ranges['pH']),
                    soil_type
                ])
        
        df = pd.DataFrame(data, columns=['N', 'P', 'K', 'pH', 'SoilType'])
        df.to_csv(SOIL_DATA_PATH, index=False)
        logger.info(f"Saved synthetic soil data to {SOIL_DATA_PATH}")
        return df
    
    except Exception as e:
        logger.error(f"Error in soil data preparation: {str(e)}")
        raise

def train_soil_model():
    """Train and save soil classification model"""
    try:
        logger.info("\n" + "="*60)
        logger.info("Starting Soil Classification Model Training")
        logger.info("="*60)
        
        # Load or generate data
        df = load_or_generate_soil_data()
        
        # Preprocessing
        X = df[['N', 'P', 'K', 'pH']]
        y = df['SoilType']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Calculate class weights
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        
        # Model architecture
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=SEED,
            n_jobs=-1
        )
        
        # Training
        logger.info("Training soil classification model...")
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info("\nSoil Model Evaluation:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
        
        # Save artifacts
        with open('models/soil_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/soil_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save class mapping
        soil_types = {
            0: 'Sandy',
            1: 'Loamy',
            2: 'Clay',
            3: 'Silty',
            4: 'Peaty',
            5: 'Chalky'
        }
        with open('models/soil_types.pkl', 'wb') as f:
            pickle.dump(soil_types, f)
        
        logger.info("Soil model training completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Soil model training failed: {str(e)}")
        return False

def prepare_plant_disease_data():
    """Prepare and augment plant disease image data"""
    try:
        # Verify dataset exists
        if not os.path.exists(PLANTVILLAGE_PATH) or len(os.listdir(PLANTVILLAGE_PATH)) == 0:
            raise FileNotFoundError(
                f"PlantVillage dataset not found at {PLANTVILLAGE_PATH}\n"
                "Please download from: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset\n"
                "And extract to the data/plantvillage directory"
            )
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Training data
        train_generator = train_datagen.flow_from_directory(
            PLANTVILLAGE_PATH,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            subset='training',
            class_mode='categorical',
            seed=SEED
        )
        
        # Validation data
        val_generator = train_datagen.flow_from_directory(
            PLANTVILLAGE_PATH,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            subset='validation',
            class_mode='categorical',
            seed=SEED
        )
        
        # Get class information
        class_names = list(train_generator.class_indices.keys())
        class_counts = np.unique(train_generator.classes, return_counts=True)
        
        logger.info(f"Found {len(class_names)} plant disease classes")
        logger.info(f"Class distribution: {dict(zip(class_names, class_counts[1]))}")
        
        return train_generator, val_generator, class_names
    
    except Exception as e:
        logger.error(f"Error preparing plant disease data: {str(e)}")
        raise

def build_disease_model(num_classes):
    """Build CNN model for plant disease classification"""
    model = models.Sequential([
        # Convolutional Base
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Custom learning rate
    optimizer = optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def train_disease_model():
    """Train and save plant disease classification model"""
    try:
        logger.info("\n" + "="*60)
        logger.info("Starting Plant Disease Model Training")
        logger.info("="*60)
        
        # Prepare data
        train_generator, val_generator, class_names = prepare_plant_disease_data()
        
        # Calculate class weights
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights = dict(enumerate(class_weights))
        
        # Build model
        model = build_disease_model(len(class_names))
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'models/disease_model_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                verbose=1
            ),
            CSVLogger(
                'training_logs/disease_training_log.csv',
                append=True
            )
        ]
        
        # Training
        logger.info("Training plant disease model...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save('models/disease_model.h5')
        
        # Save class names
        with open('models/disease_classes.pkl', 'wb') as f:
            pickle.dump(class_names, f)
        
        logger.info("Plant disease model training completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Plant disease model training failed: {str(e)}")
        return False

def main():
    """Main training function"""
    try:
        start_time = datetime.now()
        logger.info(f"\n{'='*80}")
        logger.info(f"Agricultural AI Model Training - Started at {start_time}")
        logger.info(f"{'='*80}\n")
        
        setup_directories()
        
        # Train soil model
        soil_success = train_soil_model()
        
        # Train disease model
        disease_success = train_disease_model()
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n{'='*80}")
        logger.info("Training Summary:")
        logger.info(f"- Soil Model: {'SUCCESS' if soil_success else 'FAILED'}")
        logger.info(f"- Disease Model: {'SUCCESS' if disease_success else 'FAILED'}")
        logger.info(f"- Total Duration: {duration}")
        logger.info(f"{'='*80}\n")
        
        if not soil_success or not disease_success:
            raise RuntimeError("One or more models failed to train")
        
    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)