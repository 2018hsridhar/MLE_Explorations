# import matplotlib.pyplot as plt

# # sklearn as free ML library in Python :-)
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# Set up a seperate notebook for loading datasets and slicing specific columnar features from the datasets
# Fast columnar operations capabilities?

# Run script command
# python datasetsLinearRegression.py


from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

api = KaggleApi()
api.authenticate()

# Download and unzip the dataset
target_dataset = 'fedesoriano/electric-power-consumption'
target_path = "./data"

# Download latest version
api.dataset_download_files(target_dataset,path=target_path,unzip=True)

# Read the dataset
DELIM = ';'
df = pd.read_csv('./data/powerconsumption.csv',sep=DELIM)
print(df.head())



