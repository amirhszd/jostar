import numpy as np
from ..metrics._metrics import cross_validate_LOO, cross_val_predict_loo
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.base import is_regressor
from sklearn.model_selection import cross_val_predict
import warnings
from sklearn.metrics import make_scorer

def calculate_fitness(model,cv,scorer, x,y,x_test = None, y_test = None, n_jobs = -1):
    '''
    This function calculates fitness of the function (loss/accuracy measures).
    '''
    
    n_features = x.shape[1] 
    if cv is not None:
        if (x_test is not None) and (y_test is not None):
            raise ValueError("Either test partition (x_test and y_test) or cv should be defined.")            
        if scorer.__name__ == "adj_r2_score":
            raise ModuleNotFoundError("adj_r2_score is not availale with CV.")                 
        
        # if the model is a regressor
        if is_regressor(model):                                
            if cv.__str__() == "LeaveOneOut()":
                y_pred = cross_val_predict_loo(model,x,y,cv)                    
            else:
                y_pred = cross_val_predict(model,x,y, n_jobs = n_jobs, cv=cv)
            score = scorer(y,y_pred)                            
                              
        # if the model is a classifier                
        else:
            if cv.__str__() == "LeaveOneOut()":
                y_pred = cross_val_predict_loo(model,x,y,cv)                       
                                            
            else:
                y_pred = cross_val_predict(model,x,y,n_jobs = n_jobs, cv=cv)     
            score = scorer(y,y_pred)            
        
        
    elif cv is None:        
        # if regression
        if is_regressor(model):    
            if (x_test is not None) and (y_test is not None):
                model.fit(x,y)
                y_pred = model.predict(x_test)                  
            else:
                #warnings.warn("Neither cv_model nor test partition (x_test/y_test) are defined, predicting on x/y.")
                model.fit(x,y)
                y_pred = model.predict(x)   
                y_test = y 
            score = scorer(y_test,y_pred)    
            
        # if classification                    
        else:
            if (x_test is not None) and (y_test is not None):
                model.fit(x,y)
                y_pred = model.predict(x_test)                   
            else:
                #warnings.warn("Neither cv_model nor test partition (x_test/y_test) are defined, predicting on x/y.")
                model.fit(x,y)
                y_pred = model.predict(x)       
                y_test = y
            score = scorer(y_test,y_pred)                          
             
    return (score, n_features)            