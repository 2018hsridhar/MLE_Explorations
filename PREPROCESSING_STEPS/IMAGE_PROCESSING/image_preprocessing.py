from sklearn import logger
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Goal : Properly format and augment images for ML model training

Best practices for image preprocessing:
Always normalize pixel values to [0,1] or use standard normalization
Use tf.data.AUTOTUNE for parallel processing optimization
Apply augmentation only to training data, not validation/test
Cache datasets when possible to avoid repeated preprocessing
Use prefetch to overlap data loading with model training
Resize images consistently to match model input requirements
Handle different aspect ratios appropriately (pad vs. crop vs. stretch)
'''

def imagePreProcessingPipeline():
    logger.info(f"Image preprocessing pipeline initiated.")
    # for each image in dataset
    # process each image

# Load and decode image
def load_and_preprocess_image(path):
    # Read image file
    image = tf.io.read_file(path)
    
    # Decode image (JPEG, PNG, etc.)
    numChannels = 3 # RGB   
    image = tf.image.decode_image(image, channels=numChannels)
    
    # Convert to float32 and normalize to [0,1]
    image = tf.cast(image, tf.float32) / 255.0
    
    return image

def test_image_preprocessing():
    # Example image path
    image_path = "path/to/image.jpg"
    
    # Load and preprocess image
    processed_image = load_and_preprocess_image(image_path)
    
    # Display the image
    plt.imshow(processed_image)
    plt.axis('off')
    plt.show()

# Example usage
# image_path = "path/to/image.jpg"
# processed_image = load_and_preprocess_image(image_path)