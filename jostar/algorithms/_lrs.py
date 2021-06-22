# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:56:58 2020

@author: Amirh
"""
from ..utils._base_feature_selector import BaseFeatureSelector
from ..functions import ff
import numpy as np
import time
import itertools
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.base import clone
import random


class PlusLMinusR(BaseFeatureSelector):
    def __init__(self, model, n_f, weight, scoring, l=3, r=2, cv=None,
                 verbose= True, random_state=None, n_jobs = 1,**kwargs):
        """
        PLus-L Minus-R is a sequential algorithm that iteratively adds L
        features and removes R features from the selection, using sequential
        forwards selection (SFS) and sequential backward selection (SBS). This 
        algorithm includes forward and backaward sequential selection. If L is bigger 
        R, the algorithm starts with SFS and adds L feature then removes R features
        using SBS. If R is bigger than L, the algorithm starts with SBS and removes
        R features then adds L features using SFS.
        
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
            Callable sklearn score or customized score function that takes in y_pred and y_true                                                 
        l : int, optional
            Number of features to add using sequential forward selection (SFS).
            The default is 3.
        r : int, optional
            Number of features to remove using sequential backward selection (SBS).
            The default is 2.
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
        if l != int(l):
            raise ValueError("l should be an integer")
        if r != int(r):
            raise ValueError("r should be an integer")
        if weight != int(weight):
            raise ValueError("wieght should be an integer either -1 or +1")
        try:
            self.fitness = ff
        except:
            raise ImportError("Cannot find/import ff.py defined in the same directory")
        
        self.model = model
        self.n_f = n_f
        self.cv = cv      
        self.weight = weight  
        self.scoring = scoring 
        self.l = l
        self.r = r       
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
        if self.l == self.r:
            raise ValueError('L and R cannot be euqal.')
        
        elif self.l > self.r:  
            """
            starting with an empty list and consequently running SFS and SBS
            """
            self.sel_features = []
            sel_features = []
            features_available = list(range(self.x.shape[1]))
            self.best_fits = []
            
            while len(self.sel_features) != self.n_f:
                # running Sequential Forward Selection L times
                # finding the first best feature           
                new_indices = [sel_features + [j] for j in features_available]
                if len(self.sel_features) == 0:
                    fitness_values = self._pool_fitness_calc_single(features_available) 
                else:
                    fitness_values = self._pool_fitness_calc_multiple(new_indices)
                    
                for i in range(self.l):
                    if self.weight == 1:
                        attr_index = np.argmax(fitness_values)
                    elif self.weight == -1:
                        attr_index = np.argmin(fitness_values)

                    # appending the feature to the selected feature list
                    #sel_features.extend([z for z in new_indices[attr_index] if z not in sel_features])
                    sel_features = new_indices[attr_index]
                    
                    # deleting the best found feature from the feature indices set
                    #features_available = [z for z in features_available if z not in new_indices[attr_index]]
                    features_available.remove(sel_features[-1])
                                        
                    # if last iteration, dont calculate and go to SBS
                    if i != self.l-1:
                        new_indices = [sel_features + [j] for j in features_available]
                        if len(new_indices[0]) != 1:
                            fitness_values = self._pool_fitness_calc_multiple(new_indices)    
                        else:
                            fitness_values = self._pool_fitness_calc_single(new_indices)                            
    
                # running Sequential Backward selection R times                
                # getting every combination of features available
                new_indices = list(itertools.combinations(sel_features,len(sel_features)-1))
                if len(new_indices[0]) != 1:
                    fitness_values = self._pool_fitness_calc_multiple(new_indices)                               
                else:
                    fitness_values = self._pool_fitness_calc_single(new_indices)                                                
            
                for i in range(self.r):                    
                    if self.weight == 1:
                        attr_index = np.argmax(fitness_values)
                    elif self.weight == -1:
                        attr_index = np.argmin(fitness_values)
                        
                    # appending the feature to the selected feature list so it can be identified later
                    features_available.extend([z for z in sel_features if z not in new_indices[attr_index]])
                    
                    # updating sel_features
                    sel_features = list(new_indices[attr_index])
                    
                    # if last iteration, dont calculate nad go to SFS
                    if i != self.r-1:
                        # recalculating fitness of every combination of remaining feature sets
                        new_indices = list(itertools.combinations(sel_features,len(sel_features)-1))
                        
                        if len(new_indices[0]) != 1:
                            fitness_values = self._pool_fitness_calc_multiple(new_indices)                                                
                        else:
                            fitness_values = self._pool_fitness_calc_single(new_indices)                                                   
    
                self.sel_features = sel_features                
                self.sel_features_acc = fitness_values[attr_index]
                if self.verbose:                
                    print("Features: {}  Fitness_score: {:.4} ".format(self.sel_features,self.sel_features_acc))  
                self.best_fits.append(self.sel_features_acc)                                                
                
                if len(self.sel_features) == self.l:
                    self.sel_features = self.sel_features[:self.n_f]
                        
    
        elif self.l < self.r:
            """
            starting with a full list and consequently running SBS and SFS
            """            

            self.sel_features = list(range(self.x.shape[1]))
            sel_features = list(range(self.x.shape[1]))
            features_available = []
            self.best_fits = []
            
            while len(self.sel_features) != self.n_f:               
                # running Sequential Backward Selection R times
                # removing one feature and calculating fitness values using combination      
                new_indices = list(itertools.combinations(self.sel_features,len(self.sel_features)-1))   
                fitness_values = self._pool_fitness_calc_multiple(new_indices)                                                                  
                
                for i in range(self.r):
                    # the worst feature is the one that when removed, the smallest drop in accuracy is seen                    
                    if self.weight == 1:
                        attr_index = np.argmax(fitness_values)
                    elif self.weight == -1:
                        attr_index = np.argmin(fitness_values)
                        
                    # appending the feature to the selected feature list so it can be identified later
                    features_available.extend([z for z in sel_features if z not in new_indices[attr_index]])
                    
                    # updating sel_features
                    sel_features = list(new_indices[attr_index])

                    
                    # if last iteration, dont calculate and go to SFS
                    if i != self.r-1:
                        # recalculating fitness of every combination of remaining feature sets
                        new_indices = list(itertools.combinations(sel_features,len(sel_features)-1))
                        
                        if len(new_indices[0]) != 1:
                            fitness_values = self._pool_fitness_calc_multiple(new_indices)   
    
                        else:
                            fitness_values = self._pool_fitness_calc_single(new_indices)   
    
                # running Sequential Forward Selection L times                
                new_indices = [sel_features + [j] for j in features_available]
                if len(new_indices[0]) != 1:
                    fitness_values = self._pool_fitness_calc_multiple(new_indices)   
                else:
                    fitness_values = self._pool_fitness_calc_single(new_indices)   
                                     
                for i in range(self.l):
                    if self.weight == 1:
                        attr_index = np.argmax(fitness_values)
                    elif self.weight == -1:
                        attr_index = np.argmin(fitness_values)

                    # appending the feature to the selected feature list
                    #sel_features.extend([z for z in new_indices[attr_index] if z not in sel_features])
                    sel_features = new_indices[attr_index]
                    
                    # deleting the best found feature from the feature indices set
                    #features_available = [z for z in features_available if z not in new_indices[attr_index]]
                    features_available.remove(sel_features[-1])
                    
                    # if last iteration, dont calculate and go to SBS
                    if i != self.l-1:
                        new_indices = [sel_features + [j] for j in features_available]
                        if len(new_indices[0]) != 1:
                            fitness_values = self._pool_fitness_calc_multiple(new_indices)     
                        else:
                            fitness_values = self._pool_fitness_calc_single(new_indices)   
   
                self.sel_features = sel_features
                self.sel_features_acc = fitness_values[attr_index]
                if self.verbose:
                    print("Features: {}  Fitness_score: {:.4} ".format(self.sel_features,self.sel_features_acc))  
                self.best_fits.append(self.sel_features_acc)                                                
                             
                if len(self.sel_features) == self.r:
                    self.sel_features = self.sel_features[:self.n_f]     
        
        return self.best_fits

    @property
    def _name_(self):
        return "PlusLMinusR"        