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
Preprocessing, formatting, and augmentation of images for ML model training
Model generalizability into the real world : test and validation.

Best practices for image preprocessing:
Always normalize pixel values to [0,1] or use standard normalization
Use tf.data.AUTOTUNE for parallel processing optimization
Apply augmentation only to training data, not validation/test
Cache datasets when possible to avoid repeated preprocessing
Use prefetch to overlap data loading with model training
Resize images consistently to match model input requirements
Handle different aspect ratios appropriately (pad vs. crop vs. stretch)
'''

class ImagePreprocessorLayers:

    def __init__(self):
        self.processing_layers = []
        self.augmentation_layers = []
        self.model = Sequential(self.processing_layers + self.augmentation_layers, name="image_preprocessing_pipeline")
        self.name = "image_preprocessing_pipeline"

    # height, width factor of 0.9 means zoom in by 10%  (90% of original size)
    def create_random_crop_layer(self, input_height=180, input_width=180, crop_factor=0.9):
        """
        Create a TensorFlow layer that randomly crops images
        Random cropping is a common data augmentation technique to improve model generalization
        
        Args:
            input_height: Expected input image height
            input_width: Expected input image width
            crop_factor: Factor to crop (0.9 = 90% of original size, zoom in by 10%)
        """
        crop_height = int(input_height * crop_factor)  # 90% of input height
        crop_width = int(input_width * crop_factor)    # 90% of input width
        
        crop_layer = tf.keras.layers.RandomCrop(
            height=crop_height,  # Crop height (90% of original)
            width=crop_width,    # Crop width (90% of original)
            name='random_crop'
        )
        return crop_layer

    def create_input_layer(self, input_size):
        """
        Create a TensorFlow input layer for images
        # Input shape: (batch_size, height, width, channels)
        # Variable batch size : any # of images
        # Variable height and width for input flexibility
        """
        input_layer = tf.keras.Input(shape=(270,270,3), name='input_image')  # Height and width can be variable
        return input_layer

    # TensorFlow layer for resizing images to 180x180x3
    def create_resize_layer(self, target_height=180, target_width=180):
        """
        Create a TensorFlow layer that resizes images to specified dimensions
        Default: 180x180x3 (preserves 3 RGB channels)
        
        Args:
            target_height: Target height for resized images (default: 180)
            target_width: Target width for resized images (default: 180)
            
        Note: The 3rd dimension (channels) is preserved automatically
        """
        resize_layer = tf.keras.layers.Resizing(
            height=target_height, 
            width=target_width,
            interpolation='bilinear',  # Options: 'bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', 'lanczos5', 'gaussian', 'mitchellcubic'
            crop_to_aspect_ratio=False,  # If True, crops to aspect ratio before resizing
            name=f'resize_{target_height}x{target_width}x3'  # Reflects output dimensions
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
            mode='horizontal_and_vertical',  # Options: 'horizontal', 'vertical', 'horizontal_and_vertical'
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
    
    def create_random_translation_layer(self):
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
    
    def __init__(self, target_height=180, target_width=180, augment=False):
        self.numChannels = 3 # RGB
        self.target_height = target_height
        self.target_width = target_width
        self.augment = augment
        self.preprocessor_layers = ImagePreprocessorLayers()
        self.name = f"image_preprocessing_pipeline_{target_height}x{target_width}"

    # Load and decode image
    def loadMultiFormat(self, path):
        # Read image file
        rawImage = tf.io.read_file(path)
        
        # Decode image (JPEG, PNG, etc.)
        
        # Try to decode as different formats
        try:
            # Try JPEG first
            image = tf.image.decode_jpeg(rawImage, channels=self.numChannels)
        except Exception as e:
            logger.error(f"Error decoding JPEG image: {e}")
            try:
                # Try PNG
                image = tf.image.decode_png(rawImage, channels=self.numChannels)
            except:
                # Use generic decoder
                try:
                    image = tf.image.decode_image(rawImage, channels=self.numChannels)
                except Exception as e:
                    logger.error(f"Error decoding image: {e}")
                    raise e

        # Convert to float32 for processing
        image = tf.cast(image, tf.float32)
        return image
    
    # Combine layers based on whether augmentation is enabled
    # Processing versus augmentation layers : the difference is that
    # processing layers are always applied, while augmentation layers are only applied if augment=True
    def createImagePreProcessingPipeline(self):
        logger.info(f"In function call createImagePreProcessingPipeline())")
        input_layer = self.preprocessor_layers.create_input_layer()
        resize_layer = self.preprocessor_layers.create_resize_layer(self.target_height, self.target_width)
        rescale_layer = self.preprocessor_layers.create_rescale_layer()
        flip_layer = self.preprocessor_layers.create_flip_layer()
        rotation_layer = self.preprocessor_layers.create_rotation_layer()
        contrast_layer = self.preprocessor_layers.create_contrast_layer()
        zoom_layer = self.preprocessor_layers.create_zoom_layer()
        translation_layer = self.preprocessor_layers.create_random_translation_layer()
        brightness_layer = self.preprocessor_layers.create_random_brightness_layer()
        self.processing_layers = [input_layer, resize_layer, rescale_layer]
        self.augmentation_layers = [flip_layer, rotation_layer, contrast_layer, zoom_layer, translation_layer, brightness_layer]
        self.layers = self.processing_layers + self.augmentation_layers
        self.sequentialModel = Sequential(self.layers, name=self.name)
        return self.sequentialModel

    def preprocessImage(self, image):
        imageSequentialModel = self.createImagePreProcessingPipeline()
        
        # Print model details
        print("\n" + "="*50)
        print("IMAGE PREPROCESSING MODEL DETAILS")
        print("="*50)
        self.printModelDetails(imageSequentialModel)
        print("="*50)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        outputImage = imageSequentialModel(image)
        
        # Remove batch dimension if we added it
        if outputImage.shape[0] == 1:
            outputImage = tf.squeeze(outputImage, 0)
            
        return outputImage

    def printModelDetails(self, model):
        """
        Print details about the preprocessing model
        """
        print(f"Model: {model.name}")
        print(f"Number of layers: {len(model.layers)}")
        for i, layer in enumerate(model.layers):
            print(f"  {i+1}. {layer.name} ({type(layer).__name__})")
        try:
            model.summary()
        except:
            print("Model summary not available - model may not be built yet")

    def test_image_preprocessing(self):
        # Example image path
        image_path = "../MACHINE_LEARNING_ENGINEERING_SIDE_PROJECTS/PREPROCESSING_STEPS/IMAGE_PROCESSING/IMAGES/doggo.jpg"
        
        try:
            # Load and preprocess image
            print(f"Loading image from: {image_path}")
            loaded_image = self.loadMultiFormat(image_path)
            print(f"Original image shape: {loaded_image.shape}")
            
            # # Preprocess the image
            processed_image = self.preprocessImage(loaded_image)
            print(f"Processed image shape: {processed_image.shape}")
            print(f"Processed image value range: [{tf.reduce_min(processed_image):.3f}, {tf.reduce_max(processed_image):.3f}]")
            
            # Display the processed image
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(tf.cast(loaded_image, tf.uint8))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            print("Testing with dummy data instead...")
            
            # Create dummy image data
            dummy_image = tf.random.uniform((427, 640, 3), maxval=255, dtype=tf.float32)
            print(f"Dummy image shape: {dummy_image.shape}")
            
            processed_image = self.preprocessImage(dummy_image)
            print(f"Processed dummy image shape: {processed_image.shape}")
            print(f"Processed image value range: [{tf.reduce_min(processed_image):.3f}, {tf.reduce_max(processed_image):.3f}]")
        except Exception as e:
            print(f"Error during image processing: {e}")
            import traceback
            traceback.print_exc()

def main():
    preprocessor = ImagePreprocessor(target_height=180, target_width=180, augment=True)
    preprocessor.test_image_preprocessing()

if __name__ == "__main__":
    main()