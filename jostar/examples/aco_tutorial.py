# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:19:12 2020

@author: Amirh
"""
from jostar.algorithms import NSGA2
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV

#%%
def objective(args):  

    n_f = args["n_f"]              
    n_ant = args["n_ant"]        
    Q = args["Q"]        
    tau0 = args["tau0"]          
    beta = args["beta"]     
    rho = args["rho"]     
    
    data = pd.read_csv(r"F:\SnapbeanSummer2020\regression_data_v2.csv").to_numpy()
    x = data[:,1:]
    y = data[:,0]
    model = PLSRegression()
    cv = KFold()

    # # optimizing 
    aco = ACO(model=model, n_f=n_f, weight=1, scoring='r2', cv=cv , n_iter=30, n_ant = n_ant, Q=Q, tau0=tau0, 
                      alpha=1, beta=beta, rho=rho,random_state=42)

    aco.load(x,y,decor=0.9, scale = True)
    aco.optimize()
    best_fit = aco.best_sol_acc
    
    return -best_fit

# from hyperopt import hp
# from hyperopt import fmin, tpe, space_eval

# space = {
#         'n_f': hp.choice('n_f', [5]),
#         'nAnt': hp.randint('nAnt', 10,100),
#         "Q":  hp.loguniform("Q",np.log(1e-2),np.log(0.99)),        
#         "tau0":  hp.loguniform("tau0",np.log(1e-2),np.log(0.99)),        
#         "beta": hp.randint('beta', 2,5),
#         "rho": hp.loguniform('rho', np.log(1e-2),np.log(0.99))
#         }

# best = fmin(objective, space, algo=tpe.suggest, max_evals=1000)
#%%
data = pd.read_csv(r"F:\SnapbeanSummer2020\regression_data_v2.csv").to_numpy()
x = data[:,1:]
y = data[:,0]
model = LinearRegression()
cv = LeaveOneOut()


# # optimizing 
aco = NSGA2(model=model, n_f=5, weight=(1,-1), scoring='r2', cv=cv, n_gen=5, n_pop=20)
aco.fit(x,y,decor=0.9, scale = True)
res = aco.display_results(index = 1)


"""
best loss: 0.7214362757306703

beta = 4
n_f = 5
nAnt = 16
Q = 0.016435036411856774
rho = 0.23636010984595326
tau0 = 0.08191518499245952
"""