# Script to illustrate the model architecture based on the project documentation.
# This file is for reference. To run, you need a large 'dataSet' and to install matplotlib and numpy.

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
# from matplotlib import pyplot as plt # Uncomment if you want visualization

# --- Configuration (Set these paths after running FoldersCreation.py) ---
TRAINING_DIR = "dataSet/trainingData"
TESTING_DIR = "dataSet/testingData"
MODEL_OUTPUT_DIR = "Models"
INPUT_SHAPE = (128, 128, 1) # Images are grayscale (1 color channel) and 128x128
NUM_CLASSES = 27 # A-Z (26) + blank (1)

# --- 1. Define the CNN Model Architecture (Layer 1) ---
def create_cnn_model(input_shape, num_classes):
    """Creates the main CNN model (Layer 1) for 27-class classification."""
    model = Sequential([
        # 1st Convolution Layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # 1st Pooling Layer (downsamples to 63x63)
        MaxPooling2D((2, 2)),

        # 2nd Convolution Layer
        Conv2D(32, (3, 3), activation='relu'),
        # 2nd Pooling Layer (downsamples to 30x30)
        MaxPooling2D((2, 2)),

        Flatten(), # Prepare for Dense layers

        # 1st Densely Connected Layer (Original Doc: 128 neurons, Dropout: 0.5)
        Dense(128, activation='relu'),
        Dropout(0.5),

        # 2nd Densely Connected Layer (Original Doc: 96 neurons)
        Dense(96, activation='relu'),

        # Final layer: Output Layer (SoftMax for multi-class probability)
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model using the Adam optimizer, as noted in the report
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# --- 2. Data Generators for Loading and Augmentation ---

# ImageDataGenerator automatically resizes the image to target_size, converts to float32,
# and performs scaling (rescale=1./255).
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    shear_range=0.1, 
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Assuming all images were captured as RGB and need to be converted to grayscale for model input
def create_generators(target_dir, is_training):
    return (train_datagen if is_training else test_datagen).flow_from_directory(
        target_dir,
        target_size=INPUT_SHAPE[:2],
        color_mode='grayscale', # Crucial: ensures the depth is 1
        batch_size=32,
        class_mode='categorical',
        shuffle=is_training
    )

# --- 3. Example Training Flow (Uncomment/Modify to run) ---
# if __name__ == '__main__':
#     print("Starting Model Training Setup...")
    
#     # Check for dataset existence
#     if not os.path.exists(TRAINING_DIR) or not os.path.exists(TESTING_DIR):
#         print("ERROR: Dataset directories not found. Run FoldersCreation.py and collect data first.")
#         exit()

#     # Get Data Generators
#     train_generator = create_generators(TRAINING_DIR, is_training=True)
#     validation_generator = create_generators(TESTING_DIR, is_training=False)

#     # Initialize and compile the model
#     model = create_cnn_model(INPUT_SHAPE, NUM_CLASSES)
#     model.summary()
    
#     # Note: You would also need separate logic for training the Layer 2 models ({D,R,U}, {T,K,D,I}, {S,M,N})
#     # as they are separate classifiers trained only on those specific classes.

#     # --- Train the model (Example, adjust epochs and steps as needed) ---
#     # history = model.fit(
#     #     train_generator,
#     #     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     #     epochs=20, # Use more epochs for better accuracy
#     #     validation_data=validation_generator,
#     #     validation_steps=validation_generator.samples // validation_generator.batch_size
#     # )

#     # --- Save the model (Uncomment after training) ---
#     # model_json = model.to_json()
#     # with open(os.path.join(MODEL_OUTPUT_DIR, "model_new.json"), "w") as json_file:
#     #     json_file.write(model_json)
#     # model.save_weights(os.path.join(MODEL_OUTPUT_DIR, "model_new.h5"))
#     # print("Main model saved successfully.")
