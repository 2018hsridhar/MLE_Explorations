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
from sklearn import linear_model

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
X = df[[df.columns[featureIndex] for featureIndex in range(len(df.columns)) if featureIndex != labelIndex]]
y = df.iloc[:, labelIndex]

# print(X.head)
# print(Y.head)

# how to convert strings to floats here though?
regr = linear_model.LinearRegression()
regr.fit(X,y)



featureListExampleOne = ["M", 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101,0.15, 15]
predictedNumAbaloneRings = regr.predict(featureListExampleOne)
print(predictedNumAbaloneRings)





