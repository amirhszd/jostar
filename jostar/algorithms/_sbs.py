# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:56:58 2020

@author: Amirh
"""
from ..utils._base_feature_selector import BaseFeatureSelector
from ..functions import ff
import itertools
import numpy as np
import time
import random
from sklearn.base import clone
from pathos.multiprocessing import ProcessingPool as Pool

class SBS(BaseFeatureSelector):
    def __init__(self, model, n_f, weight,scoring, cv=None,
                 verbose= True, random_state=None, n_jobs = 1,**kwargs):
        """
        Sequential backward selection (SBS) algorithm that removes features
        with lowest accuracies until the desired number of features is reached.
                        
        Parameters
        ----------
        model : class
            Instantiated Sklearn regression or classification estimator.
        n_f : int
            Number of features needed to be extracted.
        weight : int
            Maximization or minimization objective, for maximization use +1
            and for mimization use -1.
        scoring : callable
            Callable sklearn score or customized score function that takes in y_pred and y_true.                                                 
        cv : class, optional
            Instantiated sklearn cross-validation class. The default is None.
        random_state : int, optional
            Determines the random state for random number generation, 
            cross-validation, and classifier/regression model setting. 
            The default is None.
        n_jobs : int, optional
            Number of cores to use for multiprocessing. The default is 1, no
            multiprocessing is used.

        Returns
        -------
        Instantiated Optimziation model class.
        """            
        super().__init__(scoring=scoring, n_f = n_f,**kwargs)           
# =============================================================================
# checking the input arguments
# =============================================================================
        if n_f != int(n_f):
            raise ValueError("n_f should be an integer")
        if weight != int(weight):
            raise ValueError("wieght should be a tuple of either -1 or +1")
        try:
            self.fitness = ff
        except:
            raise ImportError("Cannot find/import ff.py defined in the same directory")
        
        self.model = model
        self.n_f = n_f
        self.cv = cv      
        self.weight = weight 
        
        self.verbose = verbose
        self.random_state = random_state      
        self.n_jobs = n_jobs
                    
        random.seed(self.random_state)        
        np.random.seed(self.random_state)   

        try:
            self.cv.random_state = self.random_state
            self.cv.shuffle = True
        except:
            pass
        
        try:
            self.model.random_state = self.random_state
        except:
            pass        

        
    def fit(self,x,y,decor=None,scale=False, test_size = None,**kwargs):            
        """
        Fit the model to input data X and target values Y, with extra optional 
        arguments for preprocessing data.
        
        Parameters
        ----------
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
            
        Attributes
        ----------
        best_fits: Vector of shape (n_iter,)
            Fitness values of each iteration.        
        best_sol: Vector of shape (n_f,)
            Indices of selected fetures after preprocessing input data.
        best_sol_acc: float
            Optimized fitness value for the identified solution.
        model_best: Class
            Regression/classification estimator given to the optimization 
            algorithm. At this stage model_best = model

        Returns
        -------
        Vector of shape (n_f,)
            Indices of selected fetures of the original input data (prior to 
            preprocessing). If no deocr is given, this equals to best_sol.
        Vector of shape (n_iter,)
            Fitness values of each iteration.

        """
        
        self.load(x,y,decor=decor,scale=scale)        
        if test_size is not None:
            self.train_test_split(test_size, random_state=self.random_state,**kwargs)

        a = time.time()
        # if x_train is defined
        if hasattr(self, 'x_train') and hasattr(self, 'x_test') and hasattr(self, 'y_train') and hasattr(self, 'y_test'):
            self._test = True
        else:
            self._test = False
            
        self.best_fits = self._optimize()
        self.best_sol = self.sel_features
        self.best_sol_acc = self.sel_features_acc
        self.best_features = self.best_sol
        self.model_best = clone(self.model)

        if self.verbose:               
            print("Optimization completed in {:.2f} seconds".format(time.time() - a))
            print("Best feature set is {}, {:.4f}".format(np.array(self.x_cols[self.sel_features]), self.sel_features_acc))
        
        return self.x_cols[self.sel_features], self.best_fits

    
    def _pool_fitness_calc_single(self,indices):
        if self._test:
            models = list(self.model for i in indices)
            cv_models = list(self.cv for i in indices)
            xs = list(self.x_train[:,i].reshape(-1,1) for i in indices)
            ys = list(self.y_train for i in indices)
            xtests = list(self.x_test[:,i].reshape(-1,1) for i in indices)
            ytests = list(self.y_test for i in indices)
            scorers = list(self.scorer for i in indices)
        
            if self.n_jobs == 1:
                fitness_values = np.array(list(map(self.fitness.calculate_fitness,models,
                                              cv_models,scorers,
                                              xs,ys,xtests,ytests)))[:,0]
            elif self.n_jobs > 1 :
                with Pool(self.n_jobs) as pool:
                    fitness_values = np.array(pool.map(self.fitness.calculate_fitness,models,
                                              cv_models,scorers,
                                              xs,ys,xtests,ytests))[:,0]
            elif self.n_jobs == -1 :
                with Pool(self.n_jobs) as pool:
                    fitness_values = np.array(pool.map(self.fitness.calculate_fitness,models,
                                              cv_models,scorers,
                                              xs,ys,xtests,ytests))[:,0]
                
        else:
            models = list(self.model for i in indices)
            cv_models = list(self.cv for i in indices)
            xs = list(self.x[:,i].reshape(-1,1) for i in indices)
            ys = list(self.y for i in indices)
            scorers = list(self.scorer for i in indices)

            if self.n_jobs == 1:
                fitness_values = np.array(list(map(self.fitness.calculate_fitness,models,
                                              cv_models,scorers,
                                              xs,ys)))[:,0]
            elif self.n_jobs > 1 :
                with Pool(self.n_jobs) as pool:
                    fitness_values = np.array(pool.map(self.fitness.calculate_fitness,models,
                                              cv_models,scorers,
                                              xs,ys))[:,0]
            elif self.n_jobs == -1 :
                with Pool() as pool:
                    fitness_values = np.array(pool.map(self.fitness.calculate_fitness,models,
                                              cv_models,scorers,
                                              xs,ys))[:,0]              
                    
        return fitness_values
    
    
    def _pool_fitness_calc_multiple(self,indices):
        if self._test:
            models = list(self.model for i in indices)
            cv_models = list(self.cv for i in indices)
            xs = list(self.x_train[:,i] for i in indices)
            ys = list(self.y_train for i in indices)
            xtests = list(self.x_test[:,i] for i in indices)
            ytests = list(self.y_test for i in indices)
            scorers = list(self.scorer for i in indices)
            
            
            if self.n_jobs == 1:
                fitness_values = np.array(list(map(self.fitness.calculate_fitness,
                                          models, cv_models,scorers,
                                          xs, ys,
                                          xtests, ytests)))[:,0]
            elif self.n_jobs > 1 :                
                with Pool(self.n_jobs) as pool:
                    fitness_values = np.array(pool.map(self.fitness.calculate_fitness,
                                              models, cv_models,scorers,
                                              xs, ys,
                                              xtests, ytests))[:,0]
            elif self.n_jobs == -1:    
                with Pool() as pool:
                    fitness_values = np.array(pool.map(self.fitness.calculate_fitness,
                                              models, cv_models,scorers,
                                              xs, ys,
                                              xtests, ytests))[:,0]                
                
        else:
            models = list(self.model for i in indices)
            cv_models = list(self.cv for i in indices)
            xs = list(self.x[:,i] for i in indices)
            ys = list(self.y for i in indices)
            scorers = list(self.scorer for i in indices)
            
            if self.n_jobs == 1:
                fitness_values = np.array(list(map(self.fitness.calculate_fitness,
                                          models, cv_models,scorers,
                                          xs, ys)))[:,0]                        
            elif self.n_jobs > 1:    
                with Pool(self.n_jobs) as pool:
                    fitness_values = np.array(pool.map(self.fitness.calculate_fitness,
                                              models, cv_models,scorers,
                                              xs, ys))[:,0]        
                
            elif self.n_jobs == -1:    
                with Pool() as pool:
                    fitness_values = np.array(pool.map(self.fitness.calculate_fitness,
                                              models, cv_models,scorers,
                                              xs, ys))[:,0]                    
            
        return fitness_values
    
    def _optimize(self):
        """
        optimize when train and test sets are avaialble
        """
        
        self.sel_features = list(range(self.x.shape[1]))
        sel_features = list(range(self.x.shape[1]))
        features_available = []
        self.best_fits = []
        
        # running Sequential Backward Selection R times
        # removing one feature and calculating fitness values using combination      
        new_indices = list(itertools.combinations(self.sel_features,len(self.sel_features)-1))   
        fitness_values = self._pool_fitness_calc_multiple(new_indices)                                                                  
        
        while len(sel_features) != self.n_f:
            # the worst feature is the one that when removed, the smallest drop in accuracy is seen                    
            if self.weight == 1:
                attr_index = np.argmax(fitness_values)
            elif self.weight == -1:
                attr_index = np.argmin(fitness_values)
                
            # appending the feature to the selected feature list so it can be identified later
            features_available.extend([z for z in sel_features if z not in new_indices[attr_index]])
            
            # updating sel_features
            sel_features = list(new_indices[attr_index])
            
            if self.verbose:            
                print("Features: {}  Fitness_score: {:.4} ".format(sel_features,fitness_values[attr_index])) 
            self.best_fits.append(fitness_values[attr_index])
                
                
            # recalculating fitness of every combination of remaining feature sets
            # dont calculate if number of features requested are found
            if len(sel_features) != self.n_f: 
                new_indices = list(itertools.combinations(sel_features,len(sel_features)-1))
                
                if len(new_indices[0]) != 1:
                    fitness_values = self._pool_fitness_calc_multiple(new_indices)   
                else:
                    fitness_values = self._pool_fitness_calc_single(new_indices)         
                            
        self.sel_features = sel_features
        self.sel_features_acc = fitness_values[attr_index]    
        return self.best_fits
        
    @property
    def _name_(self):
        return "SBS"        
        