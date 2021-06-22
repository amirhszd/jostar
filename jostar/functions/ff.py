
# import random
# import numpy as np
from sklearn.base import is_regressor
from sklearn.model_selection import cross_val_predict
from ..metrics._metrics import cross_val_predict_loo, rmspe_score
# import warnings
#from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import f1_score

def calculate_fitness(model,cv,scorer, x,y,x_test = None, y_test = None,n_jobs = 1):
    '''
    This function calculates fitness of the function (loss/accuracy measures).
    '''
          
    if cv is not None:
        if (x_test is not None) and (y_test is not None):
            raise ValueError("Either test partition (x_test and y_test) or cv should be defined.")            
        if scorer.__name__ == "adj_r2_score":
            raise ModuleNotFoundError("adj_r2_score is not availale with CV.")           
        
        # if our model is a regressor we are also calculating rmspe
        if is_regressor(model):                    
            if cv.__str__() == "LeaveOneOut()":
                y_pred = cross_val_predict_loo(model,x,y,cv)                                            
            else:
                y_pred = cross_val_predict(model,x,y,n_jobs = n_jobs, cv=cv)             
            score = scorer(y,y_pred) 
            perc = rmspe_score(y,y_pred)

        # if our model is a classifier we are also calculating F1
        else:
            if cv.__str__() == "LeaveOneOut()":
                y_pred = cross_val_predict_loo(model,x,y,cv)                                            
            else:
                y_pred = cross_val_predict(model,x,y,n_jobs = n_jobs, cv=cv)     
            score = scorer(y,y_pred)                      
            perc = f1_score(y,y_pred,average="micro")
                        
        
    elif cv is None:        
    # if regression task
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
            perc = rmspe_score(y_test,y_pred)

    # if classification task calculate f1
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
            perc = f1_score(y_test,y_pred,average= "micro")
                    
    return score, perc

