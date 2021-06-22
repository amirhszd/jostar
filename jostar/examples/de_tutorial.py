# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:19:22 2020

@author: Amirh
"""

from algorithms._sa import SA

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval
from sklearn.linear_model import LinearRegression
from scipy.stats import randint
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cross_decomposition import PLSRegression
#%%

# def objective(args):  

#     n_pop = args["n_pop"]              
#     cross_proba = args["cross_proba"]         
    
#     data = pd.read_csv(r"F:\SnapbeanSummer2020\regression_data_v2.csv").to_numpy()
#     x = data[:,1:]
#     y = data[:,0]
#     model = PLSRegression()
#     cv = None

#     # # optimizing 
#     de = DE(model=model, n_f=10, weight=1, scoring='r2', cv=cv , n_iter=20, n_pop = n_pop,
#             cross_proba = cross_proba, random_state=42)

#     de.fit(x,y,decor=('pearson',0.90), scale = False)

#     best_fit = de.best_sol_acc
    
#     return -best_fit
# #%%

# space = {
#         "n_pop": hp.randint("n_pop", 100,200),
#         "cross_proba": hp.loguniform("cross_proba",np.log(0.05),np.log(0.25))
#         }

# best = fmin(objective, space, algo=tpe.suggest, max_evals=1000)

#%%
param_dict = {"hidden_layer_sizes": randint(10,100)}
data = pd.read_csv(r"F:\SnapbeanSummer2020\regression_data_v2.csv").to_numpy()
x = data[:,1:]
y = data[:,0]
model = LinearRegression()
cv = None
# # optimizing 
opt_model = DE(model=model, n_f=5 ,weight=+1, scoring='r2', n_iter = 100, n_pop = 100, cv=cv,random_state=42)
df = opt_model.fit(x,y,decor=('pearson',0.9), scale = True)
#param_dict = {"hidden_layer_sizes": randint(10,100)}
#opt_model.tune_model_params(0, param_dict)

#%%
# # optimizing 
opt_model = DE(model=model, n_f=5 ,weight=+1, scoring='r2', n_iter = 100, n_pop = 100, cv=cv,random_state=42)
df_2 = opt_model.fit(x,y,decor=('pearson',0.9), scale = True)
#param_dict = {"hidden_layer_sizes": randint(10,100)}
#opt_model.tune_model_params(0, param_dict)
opt_model.display_results()









