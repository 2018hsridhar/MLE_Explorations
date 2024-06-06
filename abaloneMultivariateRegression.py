'''
Basic Multiple Linear Regression Example ( single scalar y ) 
    Not yet multivariate linear regression.
Import dataset as CSV from UCI Machine Learning

Goal : Given Physical Measurements, Predict Age ( Number of Rings )
Dataset is labeled
'''

'''
Key modules
'''
import pandas as pd
import numpy as np
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

# print(X.head)
# print(Y.head)

# how to convert strings to floats here though?

# Apply dataframe transformations.
# lambda function apply(...) change all values
df.iloc[:,0]=df.iloc[:, 0].apply(lambda x:x.replace("M","0"))
df.iloc[:,0]=df.iloc[:, 0].apply(lambda x:x.replace("F","1"))
df.iloc[:,0]=df.iloc[:, 0].apply(lambda x:x.replace("I","2"))
# print(df.head)
# print(df.dtypes)

# pd.to_numeric(df, errors='coerce')
# print(df.dtypes)
# pd.to_numeric(df['points'], downcast='float')
df.iloc[:,0] = pd.to_numeric(df.iloc[:,0], downcast='float')
# print(df.dtypes)

# Now apply data fit to our ML models.
labelIndex = numFeatures - 1
X = df[[df.columns[featureIndex] for featureIndex in range(len(df.columns)) if featureIndex != labelIndex]]
y = df.iloc[:, labelIndex]
multivarRegr = linear_model.LinearRegression()
multivarRegr.fit(X,y)

print("Coefficients of multilinear regression are:")
print(multivarRegr.coef_)

featureListExampleOne = [1, 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101,0.15]
# featureListExampleOne.reshape(-1, 1)
reshapedFeature = np.array(featureListExampleOne).reshape(1,-1)
predictedNumAbaloneRings = multivarRegr.predict(reshapedFeature)
print("For the given feature vector, my number of predicted abolone rings = " + str(predictedNumAbaloneRings))





