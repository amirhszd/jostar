# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:19:12 2020

@author: Amirh
"""
from algorithms._aco_v2 import ACO
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from functions._metrics import adj_r2_score
from functions.decor import decor
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from sklearn.linear_model import LinearRegression
from scipy.stats import randint
from sklearn.model_selection import LeaveOneOut
#%%
def objective(args):  
           
    nAnt = args["nAnt"]        
    Q = args["Q"]        
    tau0 = args["tau0"]          
    beta = args["beta"]     
    rho = args["rho"]     
    
    data = pd.read_csv(r"C:\Users\Amirh\Downloads\sfm_metrics_28mflight.csv").to_numpy()
    x = data[:,2:]
    y = data[:,1]
    model = PLSRegression()
    #cv = KFold()
    cv = None
    
    # # optimizing 
    aco = ACO(model=model, n_f=5, weight=1, scoring='r2', cv=cv , n_iter=20, nAnt = nAnt, Q=Q, tau0=tau0, 
                      alpha=1, beta=beta, rho=rho,random_state=42)

    aco.load(x,y, decor = 0.9, scale = True)
    aco.optimize()
    best_fit = aco.best_sol_acc
    
    return -best_fit
#%%


space = {
        'nAnt': hp.randint('nAnt', 50,200),
        "Q":  hp.loguniform("Q",np.log(1e-2),np.log(0.99)),        
        "tau0":  hp.loguniform("tau0",np.log(1e-2),np.log(0.99)),        
        "beta": hp.randint('beta', 2,5),
        "rho": hp.loguniform('rho', np.log(1e-2),np.log(0.99))
        }

best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
#%%


data = pd.read_csv(r"C:\Users\Amirh\Downloads\sfm_metrics_28mflight.csv").to_numpy()
x = data[:,2:]
y = data[:,1]
model = PLSRegression()
cv = None
#cv = LeaveOneOut()
#cv = KFold(10)
 
# # optimizing 
aco = ACO(model=model, n_f=5, weight=1, scoring='r2', cv=cv , n_iter=1000, nAnt = 124, Q=0.02401580548214845, tau0=0.21201587723743645, 
                  alpha=1, beta=4, rho=0.5585333355449467,random_state=42)
aco.load(x,y,decor = 0.9, scale = True)
aco.optimize()

param_dict = {"n_components": randint(1,5)}
aco.tune_model_params(param_dict)
aco.display_results()

"""
0.7868041189260054

beta 4
nAnt 124
Q 0.02401580548214845
rho 0.5585333355449467
tau0 0.21201587723743645
"""