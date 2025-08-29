# import matplotlib.pyplot as plt

# # sklearn as free ML library in Python :-)
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# Set up a seperate notebook for loading datasets and slicing specific columnar features from the datasets
# Fast columnar operations capabilities?

# Run script command
# python datasetsLinearRegression.py

# import as a class, not as a module

# Add the project root to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Now use absolute imports
from UTILS.DatasetLoader import DatasetLoader
from UTILS.DatasetPlotter import DatasetPlotter
from UTILS.MockDataGenerator import MockDataGenerator
from UTILS.CentralizedLogger import get_logger

import matplotlib.pyplot as plt
import os

# Create logs directory
os.makedirs('LOGS', exist_ok=True)

# Get centralized logger
logger = get_logger(__name__)

def executeTwoDimLinearRegression():

    logger.info(f"In function call executeTwoDimLinearRegression(): Starting ML Linear Regression Dataset analysis.")

    try:
        # # Download and unzip the dataset
        target_dataset = 'fedesoriano/electric-power-consumption'
        target_path = "./DATA"

        datasetLoader = DatasetLoader()
        datasetPlotter = DatasetPlotter()
        mockDataGenerator = MockDataGenerator()

        # data = datasetLoader.getDataset(target_path,target_dataset)
        # datasetLoader.print_dataset_summary_statistics(target_daaset,target_path,data)

        # datasetPlotter.plotDataset(data)
        NUM_SAMPLES = 100
        label_one = "celsiusAvgDailyTemps"
        label_two = "megaWattPowerDemand"
        # label_one = "Date Time"
        # label_two = "Power Zone One Consumption ( kilowatts)"
        mockData = mockDataGenerator.generateTwoDimMockTimeSeriesData(num_samples=100, dimLabelOne=label_one, dimLabelTwo=label_two)
        datasetLoader.print_dataset_summary_statistics("","",mockData)
        # datasetPlotter.plotDataset(mockData,label_one,label_two)
        # def plotDataset(self, data: pd.DataFrame, label_one:str, label_two:str) -> None:
    except Exception as e:
        logger.error(f"Error occurred in executeTwoDimLinearRegression() execution : {str(e)}", exc_info=True)
        raise

# Avoid other scripts from running
# The classes/functions main method
# This is a common Python idiom : scripts importable and executable

def main():
    executeTwoDimLinearRegression()

if __name__ == "__main__":
    main()