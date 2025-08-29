# import matplotlib.pyplot as plt

# # sklearn as free ML library in Python :-)
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# Set up a seperate notebook for loading datasets and slicing specific columnar features from the datasets
# Fast columnar operations capabilities?

# Run script command
# python datasetsLinearRegression.py

# import as a class, not as a module
from UTILS.DatasetLoader import DatasetLoader  # Import class from module
from UTILS.DatasetPlotter import DatasetPlotter  # Import class from module
from UTILS.MockDataGenerator import MockDataGenerator  # Import class from module
import matplotlib.pyplot as plt

def main():

    # # Download and unzip the dataset
    target_dataset = 'fedesoriano/electric-power-consumption'
    target_path = "./DATA"

    datasetLoader = DatasetLoader()
    datasetPlotter = DatasetPlotter()
    mockDataGenerator = MockDataGenerator()

    # data = datasetLoader.getDataset(target_path,target_dataset)
    # datasetLoader.print_dataset_summary_statistics(target_daaset,target_path,data)

    # datasetPlotter.plotDataset(data)
    num_samples = 100
    label_one = "celsiusAvgDailyTemps"
    label_two = "megaWattPowerDemand"
    # label_one = "Date Time"
    # label_two = "Power Zone One Consumption ( kilowatts)"
    mockData = mockDataGenerator.generateTwoDimMockTimeSeriesData(num_samples=100, dimLabelOne=label_one, dimLabelTwo=label_two)
    datasetLoader.print_dataset_summary_statistics("","",mockData)
    # datasetPlotter.plotDataset(mockData,label_one,label_two)
    # def plotDataset(self, data: pd.DataFrame, label_one:str, label_two:str) -> None:

main()