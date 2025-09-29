# Run script command
# python datasetsLogisticRegression.py
import sys
import os
from xml.parsers.expat import model

import sklearn.metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

# Create logs directory
os.makedirs('LOGS', exist_ok=True)

# Get centralized logger
logger = get_logger(__name__)

R2_THRESHOLD = 0.8
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_SAMPLES = 100

# Goal : Execute Logistic Regression on a binary classification dataset
# Leverage sigmoid function to predict binary outcomes (0 or 1) against a threshold
def executeLogisticRegression():

    logger.info(f"In function call executeLogisticRegression(): Starting ML Logistic Regression Dataset analysis.")

    try:
        # # Download and unzip the dataset
        # 1. Breast Cancer Wisconsin Dataset (Diagnostic) - 569 samples, 30 features, binary classification (malignant vs. benign)
        # Kaggle: Breast Cancer Wisconsin (Diagnostic) Data Set
        # M = malignant, B = benign
        target_write_dataset = 'uciml/breast-cancer-wisconsin-data'
        # target_read_dataset = 'uciml/breast-cancer-wisconsin-data'
        target_read_dataset = 'data'  # The actual file in the zip is data.csv  
        target_path = "./DATA"

        datasetLoader = DatasetLoader()
        datasetPlotter = DatasetPlotter()

        data = datasetLoader.getDataset(target_path,target_write_dataset, target_read_dataset)
        print(data.head())
        datasetLoader.print_dataset_summary_statistics(target_read_dataset,target_path,data)

        # Target features ( highly predictive )
        # mean radius
        # mean texture
        # mean perimeter
        # mean area
        # mean smoothness


        # Data pre-processing: Handle missing values, encode categorical variables, scale features
        data = data.drop(columns=['id'])  # Drop 'id' column
        # Select only the most useful features
        most_predictive_features = [
            'radius_mean',
            'texture_mean',
            'perimeter_mean',
            'area_mean',
            'smoothness_mean'
        ]
        # Separate features and target
        # OHE target
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})  # Encode 'diagnosis' column
        data = data[most_predictive_features + ['diagnosis']]

        # Drop missing values
        print(f"Rows before dropna: {len(data)}")
        data = data.dropna()
        print(f"Rows after dropna: {len(data)}")
        print(data.isnull().sum())  # Check for any remaining missing values

        X = data[most_predictive_features].values  # Features
        y = data['diagnosis'].values  # Target variable

        # # Feature scaling (only on X) ( not targets )
        # # Important for gradient descent convergence
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        logger.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
        # # Train Logistic Regression model
        model = sklearn.linear_model.LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"Logistic Regression Model Accuracy: {accuracy:.4f}")
        logger.info(f"Logistic Regression Model Precision: {precision:.4f}")
        logger.info(f"Logistic Regression Model Recall: {recall:.4f}")
        logger.info(f"Logistic Regression Model F1-Score: {f1:.4f}")

        # # Plot logistic regression decision boundary for first two features
        targetFeatureIndices = [0, 1]  # Using first two features for visualization
        datasetPlotter.plot_logistic_regression_decision_boundary(X_test, y_test, model, feature_indices=targetFeatureIndices)
        logger.info(f"Completed ML Logistic Regression Dataset analysis.")


    except Exception as e:
        logger.error(f"Error occurred in executeTwoDimLinearRegression() execution : {str(e)}", exc_info=True)
        raise

def main():
    executeLogisticRegression()

if __name__ == "__main__":
    main()