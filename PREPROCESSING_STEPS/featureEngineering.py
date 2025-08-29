'''
Handle missing data


Real estate division of an investment bank - you the data science persona
Exec feature engineering on the dataframe
Goal : Predict home tax liabilities

(1) Execute imputation of columns
(2) one-hot encode the cols
(3) min-max scale each col


30 minutes and done :-) 
WOOOH!!!
'''
import numpy as np
import pandas as pd

# [1] Execute imputation of averages : skip the NA cols 
def impute(df):
    toImputeCols = ['tax_liability','land_value','avg_bath_sqft','avg_room_sqft']
    for toImpute in toImputeCols:
        avg_to_impute = df[toImpute].mean(skipna=True)
        df[toImpute] = df[toImpute].fillna(avg_to_impute)

# [2] one-hot encode two columns
# Needed for feature eng : accuracy of ML models
# OHE : avoid ordering ( with single numbers ) -> but columnar expansion of dataframe :-(
# Issue : Vectorization of a column enlargens datasets -> sparsity
# choice of n = 10/20 : focus on key categories -> all else other -> dodge sparseness
# Expect slower training with OHE ( curseOfDim ) 
def ohe(df):
    years = [2007,2008,2009]
    yearStr = ['2007','2008','2009']
    for idx in range(len(years)):
        df[yearStr[idx]] = np.where(df['year'] == years[idx], 1, 0)
    unique_locations_list = list(df['location'].unique())
    unique_locations_list.sort()
    for uniqlocation in unique_locations_list:
        df[uniqlocation] = np.where(df['location'] == uniqlocation,1,0)
    df.drop(columns=['year','location'],inplace=True)

# [3] Min-max scaling of columns
def minMaxScale(df):
    targetCols = ['avg_room_sqft','avg_bath_sqft','land_value','tax_liability']
    for column in targetCols:
        colMin = df[column].min()
        colMax = df[column].max()
        df[column] = df[column].apply(lambda x: ((x-colMin)/(colMax-colMin)))

def get_features(df_houses):
    impute(df_houses)
    ohe(df_houses)
    minMaxScale(df_houses)
    return df_houses





    
