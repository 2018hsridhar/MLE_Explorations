# Quickly create mock datasets for testing purposes
# Testing localized data generation
# and testing data visualization

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np

class MockDataGenerator:

    def __init__(self):
        self.NUM_SAMPLES = 10
        self.NUM_FEATURES = 10
        self.START = '2020-01-01'

    def generateMockData(self,num_samples:int, num_features:int) -> pd.DataFrame:
        # Generate random data
        data = pd.DataFrame(
            np.random.rand(num_samples, num_features),
            columns=[f'feature_{i}' for i in range(num_features)]
        )
        return data

    # Generate mock time series data
    def generateTwoDimMockTimeSeriesData(self, num_samples:int, dimLabelOne:str, dimLabelTwo:str) -> pd.DataFrame:
        time_index = pd.date_range(start=self.START, periods=num_samples, freq='D')
        data = pd.DataFrame(
            {
                dimLabelOne: np.random.randn(num_samples),
                dimLabelTwo: np.random.randn(num_samples)
            },
            index=time_index
        )
        return data
