'''
A compilation of commonly-used ML techniques to address upstream steps in MLE pipelines. Techniques encompass
(A) Data Engineering
(B) Feature Engineering
(C) Cleansing steps / operations of a possible data engineering-esque pipeline.

'''


'''
Here
(A) Given a mock dataset with multiple rows of data ( records ) and each col being a feature space, let us do the following
- delete bad records or
- imputation of the records ( defaults or summary states )
- modularize function to execute.

Example CSV dataset URL : https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
Wooooh an all numeric dataset sans a header line.

'''

# Damn gotta learn python standard libraries.
# https://machinelearningmastery.com/load-machine-learning-data-python/#:~:text=The%20Python%20API%20provides%20the,directory%20(download%20from%20here).
import csv
import numpy as np
import pandas as pd

# Get this functionality working later :-)
# filename = 'pima-indians-diabetes.data.csv'
# rawData = loadMLRawDataStdPyLib(filename)

data =[[3,1,2],[3,4,5],[6,7,8]]
rawData = pd.DataFrame(data)
zerothCol = rawData[:][0]
firstCol = rawData[:][1]
# print(rawData)
# print(firstCol)

defaultVal = 0
mean = np.mean(zerothCol)
median = np.median(zerothCol)
# Array like values ( with input )
from scipy import stats
# No numpy way of caluclation here
modeVec = stats.mode(zerothCol)
mode = modeVec[0]
modeCount = modeVec[1]
# print("Mean,median,mode = ", mean, median, mode)
print(f"Mean = {mean} \t median = {median} \t modeVal = {mode} \t modeCount = {modeCount}.")

'''
Feature Scaling techniques
'''

'''
Normalize each column by {xmin,xmax}
Dataset columns are features F1,...,Fn
Normalize for only the columns given; not all columns are strictly numerical.
'''
def normalize(dataset,cols):
    xmin = min()
    xmax = max()
    for col in cols:
        print(col)
        
    
'''
z-score normalize
Fix data distributions : change (mean,sigma) from original values 
to (0,1). 
z = ( x - mu ) / sigma
'''
def standardize(dataset,cols):
    

# Datasets sourced from Kaggle { https://www.kaggle.com/ } 

rawDataset = [[1,2,3],[4,3,2],[1,2,3]]
normalizedDS = normalize(dataset)
print(normalizedDS)

'''
Dataset CSV loading techniques :
gaaah first thing to learn in MLE -> how the fuck to even load data

Also ensure to restart the kernel on code executions. It's not just clean code that matters.
Note : Jupyter is web browser -> must drag-and-drop.
'''

# Damn gotta learn python standard libraries.
# https://machinelearningmastery.com/load-machine-learning-data-python/#:~:text=The%20Python%20API%20provides%20the,directory%20(download%20from%20here).
import csv
import numpy as np
import pandas as pd

# Output : raw data by < indices, columns > ; select kth Col to solve mean on a col
# How to handle huge dataset file reading too?
# Woah may run into data limitations with this ( that and multimodality )
# Dataframe handles bigger dimensionality
def loadMLRawDataStdPyLib(filename):
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    # List on an iterator style -> then array-ify.
    x = list(reader)
    # Notice conv : integers -> Pythonic floats. Facilitates storage?
    data = np.array(x).astype('float')
    print(data.shape)

def loadMLRawDataPandasDataframe(filename):
    print(f"Starting read of filename {filename}")
    df = pd.read_csv(filename)
    # print(df.head(2))
    print(f"Finished read of filename {filename}")
    return df

# filename="C:\Users\haris\OneDrive\Desktop\KAGGLE_PUBLIC_DATASETS\car_prices.csv\car_prices.csv"
filename="car_prices.csv"
rawData = loadMLRawDataPandasDataframe(filename)
print(rawData)

