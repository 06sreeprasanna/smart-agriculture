import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plant_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = 'data/plantVillage'
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
SEED = 42
MODEL_SAVE_PATH = 'models/plant_disease_model.h5'
CLASS_NAMES_SAVE_PATH = 'models/plant_disease_classes.pkl'

def setup_directories():
    """Create required directories if they don't exist"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)

def prepare_data_generators():
    """Prepare data generators with augmentation"""
    try:
        # Verify dataset exists
        if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
            raise FileNotFoundError(
                f"PlantVillage dataset not found at {DATA_DIR}\n"
                "Please ensure the dataset is properly downloaded and extracted"
            )

        # Data augmentation configuration for training
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

        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            DATA_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            subset='training',
            class_mode='categorical',
            seed=SEED
        )

        # Validation data generator
        val_generator = val_datagen.flow_from_directory(
            DATA_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            subset='validation',
            class_mode='categorical',
            seed=SEED
        )

        # Get class information
        class_names = list(train_generator.class_indices.keys())
        class_counts = np.unique(train_generator.classes, return_counts=True)

        logger.info(f"Found {len(class_names)} classes in the dataset")
        logger.info("Class distribution:")
        for class_name, count in zip(class_names, class_counts[1]):
            logger.info(f"{class_name}: {count} samples")

        return train_generator, val_generator, class_names

    except Exception as e:
        logger.error(f"Error preparing data generators: {str(e)}")
        raise

def build_model(num_classes):
    """Build the CNN model architecture"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fifth convolutional block
        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flatten and dense layers
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    model.summary()
    return model

def train_model():
    """Train the plant disease classification model"""
    try:
        start_time = datetime.now()
        logger.info("\n" + "="*60)
        logger.info(f"Plant Disease Model Training - Started at {start_time}")
        logger.info("="*60)

        setup_directories()

        # Prepare data
        train_generator, val_generator, class_names = prepare_data_generators()

        # Calculate class weights
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights = dict(enumerate(class_weights))

        # Build model
        model = build_model(len(class_names))

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                MODEL_SAVE_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                verbose=1
            ),
            CSVLogger(
                'training_logs/plant_training_log.csv',
                append=True
            )
        ]

        # Training
        logger.info("Starting model training...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # Save the final model
        model.save(MODEL_SAVE_PATH)

        # Save class names
        import pickle
        with open(CLASS_NAMES_SAVE_PATH, 'wb') as f:
            pickle.dump(class_names, f)

        # Plot training history
        plot_training_history(history)

        # Training completion
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("\n" + "="*60)
        logger.info("Training Completed Successfully")
        logger.info(f"Total Duration: {duration}")
        logger.info("="*60)

        return True

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False

def plot_training_history(history):
    """Plot training and validation metrics"""
    try:
        # Create plots directory
        os.makedirs('training_plots', exist_ok=True)

        # Accuracy plot
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('training_plots/accuracy_plot.png')
        plt.close()

        # Loss plot
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_plots/loss_plot.png')
        plt.close()

        logger.info("Training plots saved to 'training_plots' directory")

    except Exception as e:
        logger.error(f"Error generating training plots: {str(e)}")

if __name__ == '__main__':
    train_model()