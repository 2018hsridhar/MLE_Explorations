import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Create and return a CNN
# Is a truck in the image?
# Keras API makes CNN

# ordering of layers
def classify_trucks():
    image_classif_cnn = make_image_classif_cnn()
    return image_classif_cnn
    
# 8 ordered layers
# Training handled : we return model only
    # 18 minutes and passed YAY :-) 
def make_image_classif_cnn():

    # Define the input shape for images (height, width, channels)
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3 # for RGB
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    optimizer = Adam(learning_rate=0.01) # common default

    # Define the model using the Sequential API
    # Relu = std activation function
    # seq : Conv2D -> MaxPooling2D
    model = Sequential([
        # First convolutional layer
        # Only first layer has input shape
        Conv2D(16, (3, 3), activation='relu', padding='valid',
               kernel_initializer='he_normal',
               input_shape=input_shape),
    
        # Max pooling layer
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
    
        # Second convolutional layer
        Conv2D(32, (3, 3), activation='relu', 
               kernel_initializer='he_normal',
               padding='valid'),
    
        # Max pooling layer
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
    
        # Third convolutional layer
        Conv2D(64, (3, 3), activation='relu', 
               kernel_initializer='he_normal', padding='valid'),
    
        # Flatten layer
        Flatten(),
    
        # Fully connected layer
        Dense(20, activation='relu', kernel_initializer='he_normal'),
    
        # Output layer for binary classification
        # Glorot ( Xavier ) normal initialization
        # Stable grads
        Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')
    ])
    
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(),
        metrics=['accuracy'],
    )
    return model
