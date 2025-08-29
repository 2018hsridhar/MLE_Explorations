from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import matplotlib.pyplot as plt

class DatasetPlotter:

    def __init__(self):
        self.SAMPLE_SIZE = 100
        self.RANDOM_STATE = 42

    def plotDataset(self, data: pd.DataFrame) -> None:
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
            datetime_col = sampled_data.columns[0]  # Assuming first column is datetime
            power_zone_one_col = sampled_data.columns[6]     # Assuming second column is power consumption
            xLabel = "Date Time"
            yLabel = "Power Zone One Consumption ( kilowatts)"
            title = "Electric Power Consumption Over Time"

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(sampled_data[datetime_col], sampled_data[power_zone_one_col], label=f'{power_zone_one_col}')
            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.title(title)
            plt.legend()
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.tight_layout()       # Adjust layout to prevent label cutoff
            plt.show()
        else:
            print("Dataset doesn't have enough columns for plotting")
        