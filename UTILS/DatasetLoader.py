# You need a virtual environment and the rest of that going on here 
# IDK why
# pip install kaggle
# pip install pandas
# pip install matplotlib
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

class DatasetLoader:

    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()
        self.COMMA = ','

    def getDataset(self, target_path, target_dataset) -> pd.DataFrame:
        self.api.dataset_download_files(target_dataset, path=target_path, unzip=True)
        df = pd.read_csv(f'{target_path}/{target_dataset}.csv', sep=self.COMMA)
        return df

    def print_dataset_summary_statistics(self,target_dataset,target_path,df) -> None:
        print("DATASET SUMMARY STATISTICS")
        if(target_path == None or target_dataset == None or df.empty):
            print("In function print_dataset_summary_statistics():  Invalid input parameters.")
            return
        if(target_dataset is not None and target_path is not None):
            print(f"Dataset name {target_dataset} loaded from Kaggle from targetPath {target_path}")
        print(f"Dimensions are : {df.shape}")
        print(f"Number of Rows: {df.shape[0]:,}")
        print(f"Number of Columns: {df.shape[1]}")
        print(f"Column names: {df.columns.tolist()}")

        print(f"First 5 rows:")
        print(df.head())

        print(f"Summary statistics for each column:")
        print(df.describe())