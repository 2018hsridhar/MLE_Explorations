# Add the project root to Python path
import sys
import os
from xml.parsers.expat import model

import sklearn.metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','..'))

# Now use absolute imports
from UTILS.DatasetLoader import DatasetLoader
from UTILS.DatasetPlotter import DatasetPlotter
from UTILS.MockDataGenerator import MockDataGenerator
from UTILS.CentralizedLogger import get_logger

import matplotlib.pyplot as plt
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Create logs directory
os.makedirs('LOGS', exist_ok=True)

# Get centralized logger
logger = get_logger(__name__)

R2_THRESHOLD = 0.8
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_SAMPLES = 100

'''
For SVMs and clear class distinction visualization with 100–200 points, you want small, well-separated datasets. On Kaggle, most datasets are much larger, but you can easily subsample or filter them. Here are some good options:

Iris Dataset

Kaggle: Iris Species
Classic for SVMs, 150 points, 3 classes, 4 features.

Let's get 2D plotting in
'''
def executeSupportVectorMachine():

    logger.info(f"In function call executeSupportVectorMachine(): Starting ML Support Vector Machine Dataset analysis.")

    try:
        # Example: Download and load the Iris dataset from Kaggle using DatasetLoader
        target_read_dataset = 'uciml/iris'  # Kaggle dataset identifier for Iris
        target_path = "./DATA"

        datasetLoader = DatasetLoader()
        datasetPlotter = DatasetPlotter()
        mockDataGenerator = MockDataGenerator()

        # Download and load the Iris dataset
        target_write_dataset = 'Iris'
        iris_data = datasetLoader.getDataset(target_path, target_read_dataset, target_write_dataset)
        # Optionally print summary statistics
        # datasetLoader.print_dataset_summary_statistics(target_read_dataset, target_path, iris_data)
        # Optionally plot the dataset (customize columns as needed)
        # datasetPlotter.plotDataset(iris_data, 'sepal_length', 'sepal_width')

        # --- The rest of your SVM code would go here, using iris_data as your DataFrame ---
        # For demonstration, let's create a simple SVM model on the Iris dataset
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report, confusion_matrix

        # # Load the Iris dataset
        # # Load popular datasets directly from sklearn
        # iris = datasets.load_iris()
        # X = iris.data
        # y = iris.target

        # Drop non-feature columns and separate features and labels
        # Check column names and their casing
        print(f"Iris data columns = {iris_data.columns.tolist()}")
        # Remove Unique identifier column if present
        iris_data = iris_data.drop(columns=['Id'])
        columns = iris_data.columns.tolist()
        features = columns[:-1]
        labels = columns[-1]
        # features = features[1:3]
        print(f"Features = {features}")
        print(f"Label = {labels}")
        print(f"Operating on {len(features)} features and {len(labels)} labels.")
        X = iris_data[features]
        y = iris_data[labels]

        # # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # # Create a SVM classifier
        # Linear Kernel: Simple and effective for linearly separable data.
        # Polynomial Kernel: Captures polynomial relationships, allows for complex decision boundaries.
        # Gaussian Kernel (RBF Kernel): Measures similarity based on distance; effective for non-linear data.
        # RBF-based Kernel: General category, with the Gaussian kernel being the most common type.

        # C := Regularization parameter, controls trade-off between achieving low training error and low testing error.
        # gamma := Kernel coefficient, defines influence of a single training example.

        # high C = low bias, high variance (overfitting)
        # high gamma = low bias, high variance (overfitting)
        # range of C : [0.1, 1, 10, 100]
        # range of gamma : ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100]
        svm_model = SVC(kernel='poly', C=1.0, gamma=1)  # You can experiment with different kernels

        # # Train the model
        svm_model.fit(X_train, y_train)

        # # Make predictions
        y_pred = svm_model.predict(X_test)

        # # Evaluate the model
        print("Classification Report:")
        # Desire F1 score as a balance of precision and recall
        # When both classes are imbalanced :-) 
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Let's interpret the results of metrics
        # high precision - low false positive rate
        # high recall - low false negative rate
        # F1-score - harmonic mean of precision and recall, useful for imbalanced classes
        # Support - number of true instances for each class in the dataset

        # Woah SVM correctlyc classified each results
        # This is expected for Iris dataset with linear kernel because
        # the classes are linearly separable in the feature space
        # and the dataset is small and well-structured.
        # Visualize decision boundaries if in 2D


    except Exception as e:
        logger.error(f"Error occurred in executeSupportVectorMachine() execution : {str(e)}", exc_info=True)
        raise

# Avoid other scripts from running
# The classes/functions main method
# This is a common Python idiom : scripts importable and executable

def main():
    executeSupportVectorMachine()

if __name__ == "__main__":
    main()