from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

import itertools

class ImageDatasetLoader:

    def __init__(self):
        try:
            self.api = KaggleApi()
            self.api.authenticate()
        except Exception as e:
            print(f"Error initializing Kaggle API: {e}")
            raise

    def download_image_dataset(self, target_path, target_dataset, force_download=False):
        """
        Download image dataset from Kaggle only if not already present,
        unless force_download is True.
        """
        os.makedirs(target_path, exist_ok=True)
        # Check if data already exists (look for any files in the directory)
        data_exists = any(os.scandir(target_path))
        if data_exists and not force_download:
            print(f"Dataset already exists in {target_path}. Skipping download.")
            return target_path
        else:
            print(f"Downloading dataset: {target_dataset} to {target_path}")
            self.api.dataset_download_files(target_dataset, path=target_path, unzip=True)
            print(f"Dataset downloaded to: {target_path}")
            return target_path

    def load_images_from_directory(self, dataset_path, image_size=(224, 224)):
        """Load images and labels from directory structure"""
        images = []
        labels = []
        class_names = []
        
        dataset_dir = Path(dataset_path)
        
        # Find subdirectories (typically class folders)
        for class_dir in dataset_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_names.append(class_name)
                
                # Use itertools.chain to combine multiple glob patterns
                image_files = itertools.chain(
                    class_dir.glob('*.jpg'),
                    class_dir.glob('*.png'),
                    class_dir.glob('*.jpeg')
                )
                for img_path in image_files:
                    try:
                        # Load and resize image
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(image_size)
                        img_array = np.array(img) / 255.0  # Normalize to [0,1]
                        
                        images.append(img_array)
                        labels.append(len(class_names) - 1)  # Use index as label
                        
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
        
        return np.array(images), np.array(labels), class_names

    def load_csv_image_dataset(self, csv_path, image_column='image_path', label_column='label'):
        """Load images when paths are listed in CSV"""
        df = pd.read_csv(csv_path)
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            try:
                img_path = row[image_column]
                label = row[label_column]
                
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(label)
                
            except Exception as e:
                print(f"Error loading image at row {idx}: {e}")
        
        return np.array(images), np.array(labels)

    def print_image_dataset_summary(self, images, labels, class_names=None):
        """Print summary statistics for image dataset"""
        print("="*50)
        print("IMAGE DATASET SUMMARY STATISTICS")
        print("="*50)
        print(f"Number of images: {len(images):,}")
        print(f"Image shape: {images[0].shape}")
        print(f"Number of classes: {len(np.unique(labels))}")
        print(f"Labels distribution: {np.bincount(labels)}")
        
        if class_names:
            print(f"Class names: {class_names}")
        
        # Show sample images
        self.show_sample_images(images, labels, class_names, n_samples=5)

    def show_sample_images(self, images, labels, class_names=None, n_samples=5):
        """Display sample images from the dataset"""
        fig, axes = plt.subplots(1, min(n_samples, len(images)), figsize=(15, 3))
        if n_samples == 1:
            axes = [axes]
            
        for i in range(min(n_samples, len(images))):
            axes[i].imshow(images[i])
            label = class_names[labels[i]] if class_names else labels[i]
            axes[i].set_title(f'Class: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()