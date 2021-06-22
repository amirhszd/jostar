# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:33:13 2020

@author: Amirh
"""

from ..utils._base_feature_selector import BaseFeatureSelector
from ..functions import ff
import numpy as np
from sklearn.base import clone
import random
import copy
import time

class _SimulatedAnnealing():
    def __init__(self, estimator,cv,scorer, nf, fitness, weight,
                 x,y,x_test= None, y_test = None,
                 MaxIt = 1000, MaxSubIt = 10, T0 = 1e-1, alpha = 0.99,
                 random_state=None,verbose = True):
        
        self.nf = nf
        self.estimator = estimator
        self.cv = cv
        self.MaxIt = MaxIt
        self.MaxSubIt = MaxSubIt
        self.T0 = T0 #initial temperature 
        self.alpha = alpha #cooling factor
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.fitness = fitness
        self.weight = weight
        
        self.verbose = verbose
        self.random_state = random_state
        self.scorer = scorer
        
        random.seed(self.random_state)        
        np.random.seed(self.random_state)        
        
    def CreateRandomSolution(self):
        #Function producing random permutation
        return np.random.permutation(self.x.shape[1])
    
    def RouletteWheelSelection(self,p):
        r = np.random.uniform(0, 1)
        c = np.cumsum(p)
        i = np.where(r <= c)[0][0]
        return i
    
    def ApplyInsertion(self,tour1):
        n = len(tour1)
        I = random.sample(range(n),2)
        i1 = I[0]
        i2 = I[1]

        if i1 < i2:
            indices = np.hstack([np.arange(i1),
                                     np.arange(i1+1,i2+1),
                                     np.array(i1),
                                     np.arange(i2+1,n)])
        else:
            indices = np.hstack([np.arange(i2+1),
                         np.array(i1),
                         np.arange(i2+1,i1),
                         np.arange(i1+1,n)])
    
        tour2 = copy.copy(tour1)[indices]
        return tour2
            
    def ApplySwap(self,tour1):
        n = len(tour1)
        I = random.sample(range(n),2)
        i1 = I[0]
        i2 = I[1]
        
        tour2 = copy.copy(tour1)
        tour2[i1], tour2[i2] = tour1[i2], tour1[i1]
        return tour2    
    
    def ApplyReversion(self,tour1):
        n = len(tour1)
        I = random.sample(range(n),2)
        
        i1 = min(I)
        i2 = max(I)
        
        tour2 = copy.copy(tour1)
        tour2[i1:i2+1] = tour1[i1:i2+1][::-1]
        return tour2
        
    def CreateNeighbor(self,tour1):
        pSwap = 0.2
        pReversion = 0.5
        pInsertion = 1 - pSwap - pReversion
        
        p = [pSwap, pReversion, pInsertion]
        
        method = self.RouletteWheelSelection(p)
        
        if method == 0:#OK
            #apply swap
            tour2 = self.ApplySwap(tour1)
        
        elif method == 1: #OK
            #apply reversion
            tour2 = self.ApplyReversion(tour1)
            
        elif method == 2:
            #apply insertion
            tour2 = self.ApplyInsertion(tour1)
    
        return tour2
    
    def FeatureRanker(self,all_agents):
        agents = [agent.position for agent in all_agents]
        fits = [agent.fit for agent in all_agents]
        percs = [agent.perc for agent in all_agents]

        if self.weight == 1:
            fits_first = np.percentile(fits,75)
            fits_second = np.percentile(fits,100)        
        elif self.weight == -1:
            fits_first = np.percentile(fits,0)
            fits_second = np.percentile(fits,25)        
            
        indices = np.where(np.logical_and((fits >= fits_first), (fits <= fits_second)))[0]
        
        top_agents = np.array(agents)[indices]
        top_perc = np.array(percs)[indices]
        return top_agents,top_perc 
    
    def generate(self):
        # create and evaluate initial solution
        #instantiating sol class

        Sol = type("Solution", (object,), {})                
        sol = Sol()
        
        # setting up a variable to store all agents 
        all_agents = []             
        
        sol.position = self.CreateRandomSolution()
        if (self.x_test is not None) and (self.y_test is not None):
            sol.fit, sol.perc = self.fitness.calculate_fitness(self.estimator,
                                                     self.cv,
                                                     self.scorer,
                                                     self.x[:,sol.position[:self.nf]],
                                                     self.y,
                                                     x_test=self.x_test[:,sol.position[:self.nf]],
                                                     y_test=self.y_test)   
            
        else:
            sol.fit, sol.perc = self.fitness.calculate_fitness(self.estimator,self.cv,
                                                     self.scorer,
                                                     self.x[:,sol.position[:self.nf]],
                                                     self.y)
        
        all_agents.append(sol)
        
        T = self.T0
        self.BestSol = Sol()
        self.BestSol.position = sol.position
        self.BestSol.fit = sol.fit
        # array to hold best cost values
        self.best_fits = np.zeros(self.MaxIt)        
        newsol = Sol()

        for it in range(self.MaxIt):
            for subit in range(self.MaxSubIt):
                # making sure we are not getting the same features twice
                newsol.position = self.CreateNeighbor(sol.position)
                
                if (self.x_test is not None) and (self.y_test is not None):
                    newsol.fit, newsol.perc = self.fitness.calculate_fitness(self.estimator,self.cv,
                                                                self.scorer,                                                                
                                                                self.x[:,newsol.position[:self.nf]],
                                                                self.y,
                                                                x_test=self.x_test[:,newsol.position[:self.nf]],
                                                                y_test=self.y_test)
                    
                else:
                    newsol.fit, newsol.perc = self.fitness.calculate_fitness(self.estimator,self.cv,
                                                                self.scorer,
                                                                self.x[:,newsol.position[:self.nf]],
                                                                self.y)

                
                # if newsol is better than sol
                if self.weight == 1:
                    if newsol.fit >= sol.fit:
                        sol.position = newsol.position
                        sol.fit = newsol.fit
                        sol.perc = newsol.perc
                    else:
                        #delta = (newsol.fit - sol.fit)/sol.fit
                        delta = (newsol.fit - sol.fit)
                        p = np.exp(delta/T)
                        if np.random.uniform(0, 1) <= p:
                            sol.position = newsol.position
                            sol.fit = newsol.fit
                            sol.perc = newsol.perc
                    # update best solution ever found
                    if sol.fit > self.BestSol.fit:
                        self.BestSol.position= sol.position
                        self.BestSol.fit= sol.fit
                        self.BestSol.perc= sol.perc
                                    
                elif self.weight == -1:
                    if newsol.fit <= sol.fit:
                        sol.position = newsol.position
                        sol.fit = newsol.fit
                        sol.perc = newsol.perc
                    else:
                        delta = (newsol.fit - sol.fit)/sol.fit
                        p = np.exp(-delta/T)
                        if np.random.uniform(0, 1) <= p:
                            sol.position = newsol.position
                            sol.fit = newsol.fit
                            sol.fit = newsol.fit
                
                    # update best solution ever found
                    if sol.fit < self.BestSol.fit:
                        self.BestSol.position= sol.position
                        self.BestSol.fit= sol.fit
                        self.BestSol.perc= sol.perc
                        
                # saving the new_sol
                all_agents.append(newsol)

            # store best cost ever found
            self.best_fits[it] = self.BestSol.fit
            
            # update temp
            T = self.alpha*T
                        
            if self.verbose:
                print("Iter: {} Features: {}  Fitness_score: {:.4} ".format(it+1, self.BestSol.position[:self.nf],self.BestSol.fit))  

        # getting top_agents
        tops = self.FeatureRanker(all_agents)       
        
        return np.array([self.BestSol.position[:self.nf], self.BestSol.fit]), self.best_fits, tops                  

            
class SA(BaseFeatureSelector):
    def __init__(self, model, n_f, weight, scoring, n_iter= 1000, n_sub_iter=100, cv=None,                 
                 t0 = 1e-1, alpha= 0.9,
                 verbose= True, random_state=None,**kwargs):
        
        """
        Simulated Annealing (SA) was introduces by PJM. Laarhoven and EHL Aarts 
        1987. SA is usually used in discrete spaces and mimics the 
        physical annealing. SA is widely used in Travelling Salesman problem. 
        
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
        n_iter : int, optional
            Maximum number of iterations. For more complex 
            problems it is better to set this parameter larger. 
            The default is 1000.
        n_sub_iter : int, optional
            Maximum number of sub iterations (number of neighbourhood solutions)
            per iteration. For more complex problems it is better to set this 
            parameter larger. 
            The default is 100.
        cv : class, optional
            Instantiated sklearn cross-validation class. The default is None.
        t0 : float, optional
            The initial temperature parameter. There is no reported default value
            for this parameter and determining the initial temperateure is an
            advance topic. It is suggested that users choose a value between 
            [0.1,500] and tune it. The defualt is 0.1.
        alpha : float, optional
            Cooling factor/ratio parameter. The value can reside [0,1] but 
            usually a value of [0.8,0.99] fits the best. The default is 0.9.
        verbose : bool, optional
            Wether to print out progress messages. The default is True.
        random_state : int, optional
            Determines the random state for random number generation, 
            cross-validation, and classifier/regression model setting. 
            The default is None.

        Returns
        -------
        Instantiated Optimziation model class.

        References:
            Park, M. W., & Kim, Y. D. (1998). A systematic procedure for 
            setting parameters in simulated annealing algorithms. Computers & 
            Operations Research, 25(3), 207-217.
        
        """            
        super().__init__(scoring=scoring, n_f = n_f,**kwargs)   
# =============================================================================
# checking the input arguments
# =============================================================================
        if n_f != int(n_f):
            raise ValueError("n_f should be an integer")
        if n_iter != int(n_iter):
            raise ValueError("n_iter should be an integer")
        if n_sub_iter != int(n_sub_iter):
            raise ValueError("n_sub_iter should be an integer")
        if weight != int(weight):
            raise ValueError("wieght should be a tuple of either -1 or +1")
        try:
            self.fitness = ff
        except:
            raise ImportError("Cannot find/import ff.py defined")
        
        self.model = model
        self.n_f = n_f
        self.cv = cv      
        self.weight = weight     
        self.n_iter = n_iter
        self.n_sub_iter = n_sub_iter    
        self.t0 = t0
        self.alpha = alpha
        
        self.verbose = verbose
        self.random_state = random_state
        
        try:
            self.cv.random_state = self.random_state
            self.cv.shuffle = True
        except:
            pass
        
        try:
            self.model.random_state = self.random_state
        except:
            pass

    def fit(self,x,y,decor=None,scale=False, test_size = None, n_repeats = 30,**kwargs):  
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
        n_repeats : int, optional
            Number of times to permute the feature for the calculation of 
            permutation importance. The default is 30.
            
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
        rankings: rankings vectors from the method proposed by Hassanzadeh et al. (2021)            

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
        if hasattr(self, 'x_train') and hasattr(self, 'x_test') and hasattr(self, 'y_train') and hasattr(self, 'y_test'):
            self.alg_opt = _SimulatedAnnealing(self.model,
                                      self.cv,
                                      self.scorer,
                                      self.n_f,
                                      self.fitness,
                                      self.weight,
                                      self.x_train,
                                      self.y_train,                                                                             
                                      x_test = self.x_test,
                                      y_test = self.y_test,
                                      MaxIt = self.n_iter, 
                                      MaxSubIt = self.n_sub_iter,
                                      T0= self.t0, 
                                      alpha= self.alpha,
                                      verbose = self.verbose,
                                      random_state = self.random_state)                                           
            
        else:
            self.alg_opt = _SimulatedAnnealing(self.model,
                                      self.cv,
                                      self.scorer,
                                      self.n_f,
                                      self.fitness,
                                      self.weight,
                                      self.x,
                                      self.y,                                                                             
                                      MaxIt = self.n_iter, 
                                      MaxSubIt = self.n_sub_iter,
                                      T0= self.t0, 
                                      alpha= self.alpha,
                                      verbose = self.verbose,
                                      random_state = self.random_state)                                           


        best_sol,self.best_fits, tops = self.alg_opt.generate()
        self.best_sol = best_sol[0]
        self.best_sol_acc = best_sol[1]       
        self.best_features = self.best_sol
        # setting the model_best = model since one may not want to run tune_param
        self.model_best = clone(self.model)

        #setting top rankings 
        self.rankings = self._feature_ranking(tops, self.x_cols, n_repeats=n_repeats, random_state= self.random_state)
        
        
        if self.verbose:
            print("Optimization completed in {} seconds".format(time.time() - a))
            print("Best individual is {}, {:.4}".format(np.array(self.x_cols[self.best_sol]), self.best_sol_acc))
        
        return self.x_cols[self.best_sol], self.best_fits  

    @property
    def _name_(self):
        return "SA"            
        