# Run script command
# python datasetsLogisticRegression.py
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

# Goal : Execute Logistic Regression on a binary classification dataset
# Leverage sigmoid function to predict binary outcomes (0 or 1) against a threshold
def executeLogisticRegression():

    logger.info(f"In function call executeLogisticRegression(): Starting ML Logistic Regression Dataset analysis.")

    try:
        # # Download and unzip the dataset
        # 1. Breast Cancer Wisconsin Dataset (Diagnostic) - 569 samples, 30 features, binary classification (malignant vs. benign)
        # Kaggle: Breast Cancer Wisconsin (Diagnostic) Data Set
        target_write_dataset = 'uciml/breast-cancer-wisconsin-data'
        target_read_dataset = 'data'  
        target_path = "./DATA"

        datasetLoader = DatasetLoader()
        datasetPlotter = DatasetPlotter()

        data = datasetLoader.getDataset(target_path,target_write_dataset, target_read_dataset)
        print(data.head())
        datasetLoader.print_dataset_summary_statistics(target_read_dataset,target_path,data)


    except Exception as e:
        logger.error(f"Error occurred in executeTwoDimLinearRegression() execution : {str(e)}", exc_info=True)
        raise

def main():
    executeLogisticRegression()

if __name__ == "__main__":
    main()