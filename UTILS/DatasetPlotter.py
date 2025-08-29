from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import matplotlib.pyplot as plt

class DatasetPlotter:

    def __init__(self):
        self.SAMPLE_SIZE = 100
        self.RANDOM_STATE = 42

    def plotDataset(self, data, label_one=None, label_two=None):
        print(f"Entered method plotDataset()")
        # Use random sampling for better statistical analysis
        # RANDOM_STATE for reproducibility
        sampled_data = data.sample(n=self.SAMPLE_SIZE, random_state=self.RANDOM_STATE)


        # Perform any additional analysis or processing on the data here
        # For example, you could visualize the data or train a model

        # Visualize the data
        # Check if expected columns exist before plotting
        if (len(sampled_data.columns) > 1):
            print(f"We have {len(sampled_data.columns)} columns to work with.")
            # Use actual column names from the dataset
            x_col = sampled_data[label_one]  # Assuming first column is datetime
            y_col = sampled_data[label_two]     # Assuming second column is power consumption
            xLabel = label_one if label_one else x_col
            yLabel = label_two if label_two else y_col
            title = "Electric Power Consumption Over Time"

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(sampled_data[x_col], sampled_data[y_col], label=f'{y_col}')
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.title(title)
            plt.legend()
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.tight_layout()       # Adjust layout to prevent label cutoff
            plt.show()
        else:
            print("Dataset doesn't have enough columns for plotting")
        