'''
Basic Multiple Regression Example
Import dataset as CSV from UCI Machine Learning

Goal : Given Physical Measurements, Predict Age ( Number of Rings )
Dataset is labeled
'''

'''
Key modules
'''
import pandas as pd

# filepath = "C:\Users\haris\OneDrive\Desktop\ML-Datasets\abalone\abalone.csv"

# df = pd.read_csv(r"C:\Users\haris\OneDrive\Desktop\ML-Datasets\abalone\abalone.csv")
# df = pd.read_csv(r"C:\Users\haris\OneDrive\Desktop\ML-Datasets\abalone\abalone.data")
df = pd.read_csv(r"abalone.csv")
# print(df.head())

# colHeaders = df.columns.values.tolist()
# print(colHeaders)
sampleSize = len(df)
numFeatures = len(df.columns) - 1
# print("Sample Size = " + str(sampleSize))
# print("Num features = " + str(numFeatures))
labelIndex = numFeatures - 1
X = df[[df.columns[i] for i in range(len(df.columns)) if i != labelIndex]]
Y = df.iloc[:, labelIndex]

# print(X.head)
# print(Y.head)



