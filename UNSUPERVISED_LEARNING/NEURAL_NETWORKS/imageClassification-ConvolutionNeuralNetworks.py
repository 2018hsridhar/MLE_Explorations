'''
Exploring the realm of image classification ( mostly using unsupervised learning techniques ).
We'll deep dive into CNNs - Convolution Neural Networks - and tinker around with neural network parameters : filter sizes, the number of layers, activation function, and loss functions.
We'll also look into common image preprocessing steps : setting up a standardized color mode, dimensionality, and orientation.

Is it possible to cluster all the photos in your phone automatically without labeling?




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

# Goal : Train a simple CNN on an image dataset from Kaggle
# We'll use a small dataset for demonstration purposes
# Discover clusters/patterns in the images without explicit labels
def train_simple_cnn():
    statusCode = 0
    try:
        logger.info(f"In function train_simple_cnn() : Starting CNN training with image data")
    
        # Download image dataset
        # Example dataset: https://www.kaggle.com/datasets/heavensky/image-dataset-for-unsupervised-clustering
        # This dataset contains images of various objects categorized into different folders.
        # Each folder represents a class label.
        target_dataset = 'heavensky/image-dataset-for-unsupervised-clustering'
        target_path = "./IMAGE_DATA"
        
        imageLoader = ImageDatasetLoader()
        
        # Download dataset
        dataset_path = imageLoader.download_image_dataset(target_path, target_dataset, force_download=False)
        logger.info(f"Image Dataset downloaded to: {dataset_path}")
        statusCode = 0

        # Load images and labels
        imageHeight = 64
        imageWidth = 64
        images, labels, class_names = imageLoader.load_images_from_directory(dataset_path, image_size=(imageWidth, imageHeight))
        images = np.array(images)
        labels = np.array(labels)
        logger.info(f"Loaded {len(images)} images with {len(class_names)} classes.")

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {X_train.shape[0]} images")
        logger.info(f"Testing set size: {X_test.shape[0]} images")
        num_classes = len(class_names)
        logger.info(f"Number of classes: {num_classes}")

        # Clustering, unsupervised learning
        




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