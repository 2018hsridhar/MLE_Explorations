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
        self.processing_layers = []
        self.augmentation_layers = []
        self.model = Sequential(self.processing_layers + self.augmentation_layers, name="image_preprocessing_pipeline")
        self.name = "image_preprocessing_pipeline"

    def createImagePreProcessingPipeline(self):
        logger.info(f"In function call createImagePreProcessingPipeline())")
        resize_layer = self.create_resize_layer()
        rescale_layer = self.create_rescale_layer()
        flip_layer = self.create_flip_layer()
        rotation_layer = self.create_rotation_layer()
        contrast_layer = self.create_contrast_layer()
        zoom_layer = self.create_zoom_layer()
        translation_layer = self.craete_random_translation_layer()
        brightness_layer = self.create_random_brightness_layer()
        self.processing_layers = [resize_layer, rescale_layer]
        self.augmentation_layers = [flip_layer, rotation_layer, contrast_layer, zoom_layer, translation_layer, brightness_layer]
        self.layers = self.processing_layers + self.augmentation_layers
        self.model = Sequential(self.layers, name=self.name)
        return self.model

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
    
    def create_rotation_layer(self):
        """
        Create a TensorFlow layer that randomly rotates images
        Random rotation is a common data augmentation technique to improve model generalization
        """
        rotation_layer = tf.keras.layers.RandomRotation(
            factor=0.1,  # Rotation factor, e.g., 0.1 means +/- 10% of 2π radians
            fill_mode='reflect',  # How to fill points outside boundaries: 'constant', 'reflect', 'wrap', 'nearest'
            interpolation='bilinear',  # Interpolation method: 'nearest', 'bilinear', 'bicubic'
            name='random_rotation'
        )
        return rotation_layer
    
    def create_contrast_layer(self):
        """
        Create a TensorFlow layer that randomly adjusts image contrast
        Random contrast adjustment is a common data augmentation technique to improve model generalization
        """
        contrast_layer = tf.keras.layers.RandomContrast(
            factor=0.1,  # Contrast adjustment factor, e.g., 0.1 means +/- 10% contrast change
            name='random_contrast'
        )
        return contrast_layer
    
    def create_zoom_layer(self):
        """
        Create a TensorFlow layer that randomly zooms into images
        Random zooming is a common data augmentation technique to improve model generalization
        """
        zoom_layer = tf.keras.layers.RandomZoom(
            height_factor=0.1,  # Zoom factor for height, e.g., 0.1 means +/- 10% zoom
            width_factor=0.1,   # Zoom factor for width, e.g., 0.1 means +/- 10% zoom
            fill_mode='reflect',  # How to fill points outside boundaries: 'constant', 'reflect', 'wrap', 'nearest'
            interpolation='bilinear',  # Interpolation method: 'nearest', 'bilinear', 'bicubic'
            name='random_zoom'
        )
        return zoom_layer
    
    def craete_random_translation_layer(self):
        """
        Create a TensorFlow layer that randomly translates images
        Random translation is a common data augmentation technique to improve model generalization
        """
        translation_layer = tf.keras.layers.RandomTranslation(
            height_factor=0.1,  # Translation factor for height, e.g., 0.1 means +/- 10% translation
            width_factor=0.1,   # Translation factor for width, e.g., 0.1 means +/- 10% translation
            fill_mode='reflect',  # How to fill points outside boundaries: 'constant', 'reflect', 'wrap', 'nearest'
            interpolation='bilinear',  # Interpolation method: 'nearest', 'bilinear', 'bicubic'
            name='random_translation'
        )
        return translation_layer
    
    def create_random_brightness_layer(self):
        """
        Create a TensorFlow layer that randomly adjusts image brightness
        Random brightness adjustment is a common data augmentation technique to improve model generalization
        """
        brightness_layer = tf.keras.layers.RandomBrightness(
            factor=0.1,  # Brightness adjustment factor, e.g., 0.1 means +/- 10% brightness change
            name='random_brightness'
        )
        return brightness_layer

class ImagePreprocessor:
    
    def __init__(self, target_height=224, target_width=224, augment=False):
        self.target_height = target_height
        self.target_width = target_width
        self.augment = augment

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

    def preprocess_image(self, image):
        model = self.createImagePreProcessingPipeline()
        outputImage = model(image)
        return outputImage

    def createImagePreProcessingPipeline(self):
        """
        Create a TensorFlow image preprocessing pipeline
        """
        inputs = tf.keras.Input(shape=(self.target_height, self.target_width, 3))
        x = inputs

    def test_image_preprocessing(self):
        # Example image path
        # Load and preprocess image
        image_path = "path/to/image.jpg"
        loaded_image = self.load_multi_format(image_path)
        processed_image = self.preprocess_image(loaded_image)

        # Display the image
        plt.imshow(processed_image)
        plt.axis('off')
        plt.show()

def main():
    preprocessor = ImagePreprocessor(target_height=180, target_width=180, augment=True)
    preprocessor.test_image_preprocessing()

if __name__ == "__main__":
    main()