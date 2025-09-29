from sklearn import logger
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

'''
Description :
Image preprocessing pipeline using TensorFlow sequential layers
Add layers to the sequential model during initialization

Goals : 
Properly format and augment images for ML model training


Best practices for image preprocessing:
Always normalize pixel values to [0,1] or use standard normalization
Use tf.data.AUTOTUNE for parallel processing optimization
Apply augmentation only to training data, not validation/test
Cache datasets when possible to avoid repeated preprocessing
Use prefetch to overlap data loading with model training
Resize images consistently to match model input requirements
Handle different aspect ratios appropriately (pad vs. crop vs. stretch)
'''

class ImagePreprocessorLayers():

    def __init__(self):


    def imagePreProcessingPipeline():
        logger.info(f"Image preprocessing pipeline initiated.")
        # for each image in dataset
        # process each image

    # TensorFlow layer for resizing images to 180x180x3
    def create_resize_layer(self):
        """
        Create a TensorFlow layer that resizes images to 180x180x3
        """
        resize_layer = tf.keras.layers.Resizing(
            height=180, 
            width=180,
            interpolation='bilinear',  # Options: 'bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', 'lanczos5', 'gaussian', 'mitchellcubic'
            crop_to_aspect_ratio=False,  # If True, crops to aspect ratio before resizing
            name='resize_180x180'
        )
        return resize_layer

    def create_rescale_layer(self):
        """
        Create a TensorFlow layer that rescales pixel values to [0,1]
        Rescaling enables faster convergence during training
        0-255 pixel values are common in images, rescaling to 0-1 range is standard practice
        1./255 scales values down, offset=0.0 means no shift after scaling
        0-1 normalized values help with numerical stability and model performance
        0-1 range is compatible with common activation functions like ReLU and sigmoid`
        """
        rescale_layer = tf.keras.layers.Rescaling(
            scale=1./255,  # Scale factor
            offset=0.0,    # Offset to add after scaling
            name='rescale_0_1'
        )
        return rescale_layer

    def create_flip_layer(self):
        """
        Create a TensorFlow layer that randomly flips images horizontally
        Random flipping is a common data augmentation technique to improve model generalization
        """
        flip_layer = tf.keras.layers.RandomFlip(
            mode='horizontal',  # Options: 'horizontal', 'vertical', 'horizontal_and_vertical'
            name='random_flip'
        )
        return flip_layer

# Alternative: Using tf.image.resize function
def resize_image_function(image, target_size=(180, 180)):
    """
    Resize image using tf.image.resize function
    """
    # Ensure image has 3 channels (RGB)
    if len(image.shape) == 3 and image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image) if image.shape[-1] == 1 else image
    
    # Resize image to target size
    resized_image = tf.image.resize(
        image, 
        size=target_size, 
        method='bilinear',  # Options: 'bilinear', 'nearest_neighbor', 'bicubic', 'area'
        preserve_aspect_ratio=False,
        antialias=False
    )
    
    # Ensure output shape is (180, 180, 3)
    resized_image = tf.ensure_shape(resized_image, [180, 180, 3])
    
    return resized_image

# Complete preprocessing model with resizing layer
def create_preprocessing_model():
    """
    Create a complete preprocessing model with resizing to 180x180x3
    """
    model = tf.keras.Sequential([
        # Resize layer
        tf.keras.layers.Resizing(180, 180, name='resize_to_180x180'),
        
        # Normalization layer (rescale to [0,1])
        tf.keras.layers.Rescaling(1./255, name='normalize'),
        
        # Optional: Data augmentation layers
        tf.keras.layers.RandomFlip('horizontal', name='random_flip'),
        tf.keras.layers.RandomRotation(0.1, name='random_rotation'),
        tf.keras.layers.RandomZoom(0.1, name='random_zoom'),
        tf.keras.layers.RandomContrast(0.1, name='random_contrast'),
    ], name='image_preprocessing_180x180')
    
    return model

# Example usage function
def preprocess_image_to_180x180(image_path):
    """
    Complete image preprocessing pipeline to 180x180x3
    """
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    
    # Convert to float32
    image = tf.cast(image, tf.float32)
    
    # Resize to 180x180x3
    image = resize_image_function(image, (180, 180))
    
    # Normalize to [0,1]
    image = image / 255.0
    
    return image

# Load and decode image
def load_multi_format(path):
    # Read image file
    rawImage = tf.io.read_file(path)
    
    # Decode image (JPEG, PNG, etc.)
    numChannels = 3 # RGB   
      
    # Try to decode as different formats
    try:
        # Try JPEG first
        image = tf.image.decode_jpeg(rawImage, channels=3)
    except:
        try:
            # Try PNG
            image = tf.image.decode_png(rawImage, channels=3)
        except:
            # Use generic decoder
            image = tf.image.decode_image(rawImage, channels=3)
    
    # Ensure shape is set
    image.set_shape([None, None, 3])
    return image

def preprocess_image(image, target_height=224, target_width=224, augment=False):

    # Define preprocessing layers
    preprocessing_layers = tf.keras.Sequential()

    # Resize layer




    # Resize image
    # image = tf.image.resize(image, [target_height, target_width])
    # Convert to float32 and normalize to [0,1]
    # image = tf.cast(image, tf.float32) / 255.0
    # return image

def test_image_preprocessing():
    # Example image path
    # Load and preprocess image
    image_path = "path/to/image.jpg"
    loaded_image = load_multi_format(image_path)
    processed_image = preprocess_image(loaded_image, augment=True)

    # Display the image
    plt.imshow(processed_image)
    plt.axis('off')
    plt.show()

# Example usage and testing for 180x180x3 resizing
def demo_180x180_resizing():
    """
    Demonstrate different ways to resize images to 180x180x3
    """
    print("Creating TensorFlow layers for 180x180x3 image resizing...")
    
    # Method 1: Using Keras Resizing layer
    resize_layer = create_resize_layer()
    print(f"Created resize layer: {resize_layer.name}")
    
    # Method 2: Using preprocessing model
    preprocessing_model = create_preprocessing_model()
    print(f"Created preprocessing model with layers: {[layer.name for layer in preprocessing_model.layers]}")
    
    # Method 3: Create a simple model that just resizes
    simple_resize_model = tf.keras.Sequential([
        tf.keras.layers.Resizing(180, 180, name='resize_180x180x3'),
        tf.keras.layers.Rescaling(1./255, name='normalize_0_1')
    ], name='simple_180x180_resizer')
    
    # Example with dummy data
    dummy_image = tf.random.uniform((1, 256, 256, 3), maxval=255, dtype=tf.float32)
    print(f"Original image shape: {dummy_image.shape}")
    
    # Resize using the layer
    resized_image = resize_layer(dummy_image)
    print(f"Resized image shape (layer): {resized_image.shape}")
    
    # Resize using the preprocessing model
    processed_image = preprocessing_model(dummy_image)
    print(f"Processed image shape (model): {processed_image.shape}")
    
    # Resize using simple model
    simple_resized = simple_resize_model(dummy_image)
    print(f"Simple resized image shape: {simple_resized.shape}")
    
    return resize_layer, preprocessing_model, simple_resize_model

# Dataset creation with 180x180x3 resizing
def create_dataset_with_180x180_resize(image_paths, labels, batch_size=32):
    """
    Create a TensorFlow dataset that automatically resizes images to 180x180x3
    """
    def preprocess_path(image_path, label):
        # Load and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        
        # Resize to 180x180x3
        image = tf.image.resize(image, [180, 180])
        
        # Ensure shape and normalize
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.ensure_shape(image, [180, 180, 3])
        
        return image, label
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    # Run the demonstration
    demo_180x180_resizing() 