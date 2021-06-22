# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:00:48 2020

@author: Amirh
"""
from sklearn.metrics import r2_score, cohen_kappa_score, mean_squared_error
import numpy as np
from sklearn.base import clone
import warnings

def adj_r2_score(y_true,y_pred=None,x=None,r2=None):
    if r2 is None:
        r2 = r2_score(y_true,y_pred)
    if x is None:
        raise ValueError("Argument x needs to be defined.")
    n = x.shape[0] #number of samples
    p = x.shape[1] #number of features
    adjusted_r_squared = 1 - (1-r2)*(n-1)/(n-p-1)
    return adjusted_r_squared 

def kappa_score(y_true,y_pred):
    return cohen_kappa_score(y_true,y_pred)

def rmse_score(y_true,y_pred):
    return mean_squared_error(y_true,y_pred,squared=False)

def rmspe_score(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))
    return loss    

def cross_val_predict_loo(model,x,y,cv):
    y_preds = np.zeros(y.shape)
    for train_idx, test_idx in cv.split(x):
        X_train, X_test = x[train_idx], x[test_idx] 
        y_train, _ = y[train_idx], y[test_idx]
        
        model_new = clone(model)
        model_new.fit(X_train, y_train) 
        y_pred = model_new.predict(X_test)
            
        y_preds[test_idx] = y_pred
            
    return y_preds

def cross_validate_LOO(model,x,y,cv,scoring):
    ytests = []
    ypreds = []
    for train_idx, test_idx in cv.split(x):
        X_train, X_test = x[train_idx], x[test_idx] 
        y_train, y_test = y[train_idx], y[test_idx]
        
        model_new = clone(model)
        model_new.fit(X_train, y_train) 
        y_pred = model_new.predict(X_test)
            
        # there is only one y-test and y-pred per iteration over the loo.split, 
        # so to get a proper graph, we append them to respective lists.            
        ytests += list(y_test)
        ypreds += list(y_pred)
    
    if scoring.__name__ == "adj_r2_score":
        #warnings.warn("Adjusted R2 is not accurate with cross-validation. Consider using another accuracy metric.")
        score = scoring(ytests, ypreds,x)
    else:
        score = scoring(ytests, ypreds)
        
    return score
        
