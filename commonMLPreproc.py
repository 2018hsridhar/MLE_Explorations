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


# Output : raw data by < indices, columns > ; select kth Col to solve mean on a col
# def loadMLRawDataStdPyLib(filename):
#     raw_data = open(filename, 'rt')
#     reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
#     # List on an iterator style -> then array-ify.
#     x = list(reader)
#     # Notice conv : integers -> Pythonic floats. Facilitates storage?
#     data = numpy.array(x).astype('float')
#     print(data.shape)
#     return data
