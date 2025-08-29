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
import matplotlib.pyplot as plt


def main():

    # # Download and unzip the dataset
    target_dataset = 'fedesoriano/electric-power-consumption'
    target_path = "./data"

    datasetLoader = DatasetLoader()

    data = datasetLoader.getDataset(target_path,target_dataset)
    # datasetLoader.print_dataset_summary_statistics(target_dataset,target_path,data)

    # Perform any additional analysis or processing on the data here
    # For example, you could visualize the data or train a model

    # Visualize the data
    # Check if expected columns exist before plotting
    if (len(data.columns) > 1):
        print(f"We have {len(data.columns)} columns to work with.")
        # Use actual column names from the dataset
        datetime_col = data.columns[0]  # Assuming first column is datetime
        power_zone_one_col = data.columns[6]     # Assuming second column is power consumption
        xLabel = "Date Time"
        yLabel = "Power Zone One Consumption ( kilowatts)"
        title = "Electric Power Consumption Over Time"

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(data[datetime_col], data[power_zone_one_col], label=f'{power_zone_one_col}')
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.legend()
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()       # Adjust layout to prevent label cutoff
        plt.show()
    else:
        print("Dataset doesn't have enough columns for plotting")


main()