# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:19:12 2020

@author: Amirh
"""
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval

#%%

class ParamTuner():
    def __init__(self, opt_model,
                 x,y,
                 decor=None, scale=False, 
                 cv=None, test_size=None):
        """
        The parameter tuning class using Hyperopt module.

        Parameters
        ----------
        opt_model : Class
            The instantiated class of optimization model.
        x : ndarray or sparse matrix of shape (n_samples, n_features)
            Input data.
        y : ndarray of shape (n_samples,)
            Target values.
        decor : tuple or float, optional
            Decorrellation parameter. If a number in range [0,1] is chosen
            then the input features that have a correllation factor above the 
            give value, are removed. If tuple of (string, float [0,1]) is chosen
            then the input data features are ranked based on their feature
            importance method, indicated by the string, then decorrelated and the
            ones with higher importance are retained. The string could be one
            from the following:                
                "variance"
                "f_test"
                "mutual_info"
                "chi2"
                "pearson"                
            The default is None.
        scale : bool, optional
            Wether to scale the input data or not (centering and scaling).
            The default is False.
        test_size : float, optional
            Represents the portion of input data to be included for test split. 
            Needs to be between 0 and 1. The default is None.

        """
        
        self.x = x
        self.y = y
        self.opt_model = opt_model # TODO make sure all the otpimizers have best_sol_acc, and make sure you could also pass NSGA2 to this
        self.cv = cv
        self.decor = decor
        self.scale = scale
        self.test_size = test_size

    def tune(self, param_dict, max_evals=1000, index = 0, obj = 0):
        """
        Function taking in the dictionary of parameters and performing 
        hyperopt tuning. 

        Parameters
        ----------
        param_dict : Dictionary
            Dictionary of parameter for the optimization algorithms.
        max_evals : int, optional
            Maximum number of evaluation. The default is 1000.
        index : int, optional
            index of the accurcaies for the NSGA2 results. If NSGA2 is being
            optimized and nothing is passed, it will choose the first index.
        obj : int, optional
            index of the objective for the NSGA2 results. If nothing is passed, first one is chosen.
        Returns
        -------
        best : Dataframe
            Dataframe of best parameters.
        """
        self.index = index
        self.obj = obj
        best = fmin(self._objective, param_dict, algo=tpe.suggest, max_evals=max_evals)
        return best 
    
    def _objective(self,args):          
        self.opt_model.__dict__.update((k, v) for k, v in args.items())    
        
        # # optimizing 
        self.opt_model.fit(self.x,self.y,decor = self.decor, scale = self.scale, test_size = self.test_size)
        if self.opt_model._name_ == "NSGA2":
            best_fit = self.opt_model.best_sols_acc[self.obj][self.index]
        else:
            best_fit = self.opt_model.best_sol_acc
        
        if self.opt_model._name_ == "NSGA2":
            if self.opt_model.weight[self.obj] == 1:
                return - best_fit
            elif self.opt_model.weight[self.obj] == -1:
                return best_fit
        else:            
            if self.opt_model.weight == 1:
                return - best_fit
            elif self.opt_model.weight == -1:
                return best_fit
    