'''
Basic Linear Regression Example
Import dataset as CSV from UCI Machine Learning
'''
import pandas as pd

# filepath = "C:\Users\haris\OneDrive\Desktop\ML-Datasets\abalone\abalone.csv"

# df = pd.read_csv(r"C:\Users\haris\OneDrive\Desktop\ML-Datasets\abalone\abalone.csv")
# df = pd.read_csv(r"C:\Users\haris\OneDrive\Desktop\ML-Datasets\abalone\abalone.data")
df = pd.read_csv(r"abalone.csv")
print(df)

