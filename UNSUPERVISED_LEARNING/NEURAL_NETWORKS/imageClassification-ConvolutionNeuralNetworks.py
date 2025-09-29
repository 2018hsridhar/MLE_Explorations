'''
Exploring the realm of image classification ( mostly using unsupervised learning techniques ).
We'll deep dive into CNNs - Convolution Neural Networks - and tinker around with neural network parameters : filter sizes, the number of layers, activation function, and loss functions.
We'll also look into common image preprocessing steps : setting up a standardized color mode, dimensionality, and orientation.
'''
# Add the project root to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))

from UTILS.ImageDatasetLoader import ImageDatasetLoader
from UTILS.CentralizedLogger import get_logger
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

# Create logs directory
os.makedirs('LOGS', exist_ok=True)

# Get centralized logger
logger = get_logger(__name__)

def train_simple_cnn():
    statusCode = 0
    try:
        logger.info(f"In function train_simple_cnn() : Starting CNN training with image data")
    
        # Download image dataset
        target_dataset = 'heavensky/image-dataset-for-unsupervised-clustering'
        # target_dataset = 'tongpinmo/cat-and-dog'  # Cats vs Dogs dataset
        target_path = "./IMAGE_DATA"
        
        imageLoader = ImageDatasetLoader()
        
        # Download dataset
        dataset_path = imageLoader.download_image_dataset(target_path, target_dataset, force_download=False)
        logger.info(f"Dataset downloaded to: {dataset_path}")
        statusCode = 0
    except Exception as e:
        logger.error(f"Error in function train_simple_cnn() : CNN training: {str(e)}", exc_info=True)
        statusCode = 1
    finally:
        return statusCode


def main():
    print(f"Running {__file__} as main script")
    statusCode = train_simple_cnn()
    if(statusCode == 0):
        print(f"Script {__file__} executed successfully")
    else:
        print(f"Script {__file__} failed with status code {statusCode}")

if __name__ == "__main__":
    main()