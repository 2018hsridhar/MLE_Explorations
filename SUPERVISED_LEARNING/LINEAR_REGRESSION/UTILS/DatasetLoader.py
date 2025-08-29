from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

class DatasetLoader:

    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()
        self.DELIM = ';'

    def getDataset(self, target_path, target_dataset) -> pd.DataFrame:
        self.api.dataset_download_files(target_dataset, path=target_path, unzip=True)
        df = pd.read_csv(f'{target_path}/powerconsumption.csv', sep=self.DELIM)
        return df

    def print_dataset_summary_statistics(self,target_dataset,target_path,df) -> None:
        print("DATASET SUMMARY STATISTICS")
        print(f"Dataset name {target_dataset} loaded from Kaggle from targetPath {target_path}")
        print(f"Number of Rows: {df.shape[0]:,}")
        print(f"Number of Columns: {df.shape[1]}")
        print(f"Column names: {df.columns.tolist()}")

        print(f"First 5 rows:")
        print(df.head())

        print(f"Summary statistics for each column:")
        print(df.describe())