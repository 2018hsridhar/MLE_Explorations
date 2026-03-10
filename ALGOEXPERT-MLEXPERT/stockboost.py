import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd

'''
Gather much daily historical stock prices with features
not(features) = (index)
Features = 

POSC - 2 moving averages in tandem - spot divergence ( reversals )
X_train paired with binary label ( buy_signal ) 
    1 = 10-day percentage >= 5%
    0 = otherwise
In y_train dataframe ( index ) 

XGBoost binClasif
'''
def stock_boost(X_train, y_train, X_test):
   
    THRESHOLD = 0.5
    
    # large dataset, mem-efficient, faster col acceess
    # needed for dTrain ( ok ) 
    # account for cat feature ( sector ) 
    dtrain = xgb.DMatrix(
        X_train,
        y_train, 
        enable_categorical=True
    )
    dtest = xgb.DMatrix(
        X_test,
        enable_categorical=True
    )
    
    # Create model with model config/def
    # Objective function?
    # boosted trees have a max depth
    # categories of sector ( specify ) 
    # native cat - hist or gpu_hist
    trainParams = {
        "objective": "binary:logistic",
        "tree_method": "hist"
    }

    model = xgb.train(trainParams, dtrain)
        
    # Predictions ( match corresponding indices again )
    # Create new dataframe ( already index col )
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > THRESHOLD).astype(int)
    pred_df = pd.DataFrame({
        "prediction": y_pred
    }, index=X_test.index)
    return pred_df
