# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:58:43 2020

@author: Amirh
"""
from ..utils._base_feature_selector import BaseFeatureSelector
from ..functions import ff
import numpy as np
import random
from sklearn.base import clone
import time

class _DifferentialEvolution():
    def __init__(self, estimator,cv, scorer, nf, fitness, weight,
                 x,y,x_test= None, y_test = None,
                 MaxIt = 1000, nPop = 10, 
                 low_bound = 0.5, upper_bound = 1, cross_proba = 0.3,
                 random_state=None,verbose = True):
        
        self.nf = nf
        self.estimator = estimator
        self.cv = cv
        self.scorer = scorer
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.fitness = fitness
        self.weight = weight    
        self.MaxIt = MaxIt
        self.nPop = nPop    
        
        self.VarSize = self.x.shape[1]
        self.VarMin = 0
        self.VarMax = 1
        
        # DE parameters
        self.beta_min = low_bound #lower bound scaling factor
        self.beta_max = upper_bound #upper bound scaling factor
        self.pCR = cross_proba      #crossover probability
        
        self.verbose = verbose
        self.random_state = random_state
        self.scorer = scorer
        
        random.seed(self.random_state)        
        np.random.seed(self.random_state)        

    def RK(self,u):
        # this function creates random keys based on continuous values
        return np.argsort(u)[:self.nf]
        #return np.argsort(u)
    
    def Mutation(self,a,b,c):
        beta = np.random.uniform(self.beta_min,self.beta_max,self.VarSize)
        mutated = self.pop[a].position + np.power(beta,self.pop[b].position - self.pop[c].position)
        return mutated
    
    def CrossOver(self,ind1,ind2):
        ind3 = np.zeros(len(ind1))
        j0 = random.randint(0,len(ind1)-1)
        
        for j in range(len(ind1)):
            if j == j0 and np.random.uniform(0, 1) <= self.pCR:
                ind3[j] = ind2[j]
            else:
                ind3[j] = ind1[j]
                
        return ind3
    
    def FeatureRanker(self,all_agents):
        agents = [self.RK(agent.position) for agent in all_agents]
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
        #instantate invdivual class
        Individual  = type('Individual', (object,), {})
        self.pop = [Individual() for i in range(self.nPop)]
        
        # best particle   
        self.BestSol = Individual()
        if self.weight == 1:
            self.BestSol.fit = -np.inf
        elif self.weight == -1:
            self.BestSol.fit = np.inf
            
        # initialize an empty newsol
        NewSol = Individual()     
        
        # setting up a variable to store all agents 
        all_agents = []                
        
        # array to save best fits
        self.best_fits = np.zeros(self.MaxIt)            
            
        # initialize initial population
        for i in range(self.nPop):
            self.pop[i].position = np.random.uniform(self.VarMin,self.VarMax,self.VarSize)           
        
            # evaluation
            if (self.x_test is not None) and (self.y_test is not None):
                self.pop[i].fit, self.pop[i].perc = self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,self.RK(self.pop[i].position)],
                        self.y,
                        x_test=self.x_test[:,self.RK(self.pop[i].position)],
                        y_test=self.y_test)
                
            else:
                self.pop[i].fit, self.pop[i].perc = self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,self.RK(self.pop[i].position)],
                        self.y)           
        
            #update global best
            if self.weight == 1 and self.pop[i].fit > self.BestSol.fit:
                self.BestSol.position = self.pop[i].position
                self.BestSol.fit = self.pop[i].fit
                self.BestSol.perc = self.pop[i].perc
                
            elif self.weight == -1 and self.pop[i].fit < self.BestSol.fit:
                self.BestSol.position = self.pop[i].position
                self.BestSol.fit = self.pop[i].fit
                self.BestSol.perc = self.pop[i].perc
        
        
        #DE main loop
        for it in range(self.MaxIt):
            for i in range(self.nPop):
                
                ind1 = self.pop[i].position
                
                A = np.random.permutation(self.nPop)
                to_del = np.where(A==i)[0][0]
                A = np.delete(A,to_del)
                
                a = A[0]
                b = A[1]
                c = A[2]
                
                # perform mutation and get a new ind
                ind2 = self.Mutation(a,b,c)
                
                # perform crossover between ind1 and ind2 and get ind3
                ind3 = self.CrossOver(ind1,ind2)
                
                NewSol.position = ind3
                
                # evaluation on ind3
                if (self.x_test is not None) and (self.y_test is not None):
                    NewSol.fit, NewSol.perc = self.fitness.calculate_fitness(self.estimator,
                                                                self.cv,
                                                                self.scorer,
                                                                self.x[:,self.RK(NewSol.position)],
                                                                self.y,
                                                                x_test=self.x_test[:,self.RK(NewSol.position)],
                                                                y_test=self.y_test)   
                    
                else:
                    NewSol.fit, NewSol.perc = self.fitness.calculate_fitness(self.estimator,
                                                                self.cv,
                                                                self.scorer,
                                                                self.x[:,self.RK(NewSol.position)],
                                                                self.y)
                
                # update pop if newsol is better than ith indivdual
                if self.weight == 1 and NewSol.fit > self.pop[i].fit:
                    self.pop[i].position = NewSol.position
                    self.pop[i].fit = NewSol.fit
                    self.pop[i].perc = NewSol.perc
                    
                    #update global best
                    if self.pop[i].fit > self.BestSol.fit:
                        self.BestSol.position = self.pop[i].position
                        self.BestSol.fit = self.pop[i].fit                
                        self.BestSol.perc = self.pop[i].perc                
                    
                elif self.weight == -1 and self.pop[i].fit < self.pop[i].fit:
                    self.pop[i].position = NewSol.position
                    self.pop[i].fit = NewSol.fit
                    self.pop[i].perc = NewSol.perc
                    
                    #update global best
                    if self.pop[i].fit < self.BestSol.fit:
                        self.BestSol.position = self.pop[i].position
                        self.BestSol.fit = self.pop[i].fit
                        self.BestSol.perc = self.pop[i].perc
                
            self.best_fits[it] = self.BestSol.fit 
            
            # updating all agents
            all_agents.extend(self.pop)            
            
            if self.verbose:
                print("Iter: {} Features: {}  Fitness_score: {:.4} ".format(it+1,self.RK(self.BestSol.position),self.BestSol.fit))
        
        # getting top_agents
        tops = self.FeatureRanker(all_agents) 
        
        return np.array([self.RK(self.BestSol.position), self.BestSol.fit]), self.best_fits, tops
                 
    
class DE(BaseFeatureSelector):
    def __init__(self, model, n_f, weight, scoring, n_iter = 1000, n_pop=20, cv=None, 
                 cross_proba = 0.3, low_bound = 0.5, upper_bound = 1, 
                 verbose= True, random_state=None,**kwargs):
        
        """
        DE or Differential Evolution is an global optimization algorithm 
        which was first introduced by Storn and Price in 1990s. 
        DE is a gradient free algorithm and does not require its objective
        to be differentiable. These properties have made DE a great candidate
        for the feature or variable selection problem.
        
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
        n_pop : int, optional
            number of population size at each iteration. Typically, this 
            parameter is set to be 10*n_f, but it is dependent on the complexity 
            of the model and it is advised that user tune this parameter based 
            on their problem. The default is 20.
        cv : class, optional
            Instantiated sklearn cross-validation class. The default is None.
        cross_proba : float, optional
            The probability of individuals or agents to mate and generate
            offspring. This value is usually chosen considerably lower than one
            (e.g., 0.3), if convergence is not being achieved one can choose 
            a higher value. The higher the value, the less change of falling
            into local optima but higher compuational time. The lower the
            value the higher chance of falling in local optima but lower
            computational time. The default is 0.3.
        low_bound : float, optional
            The lower bound for choosing beta (F) value in mutation. 
            The default is 0.5.
        upper_bound : float, optional
            The upper bound for choosing beta (F) value in mutation. 
            The default is 1.
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
            Storn, R. (1996, June). On the usage of differential evolution for 
            function optimization. In Proceedings of North American Fuzzy 
            Information Processing (pp. 519-523). IEEE.
            
        
        """            
        super().__init__(scoring=scoring, n_f = n_f,**kwargs)
        
        # checking the input arguments
        if n_f != int(n_f):
            raise ValueError("n_f should be an integer")
        if n_pop != int(n_pop):
            raise ValueError("n_pop should be an integer")
        if n_iter != int(n_iter):
            raise ValueError("n_iter should be an integer")
        if weight != int(weight):
            raise ValueError("wieght should be a tuple of either -1 or +1")
        try:
            self.fitness = ff
        except:
            raise ImportError("Cannot find/import ff.py defined")
                
        
        self.model = model
        self.n_f = n_f
        self.weight = weight     
        self.scoring = scoring 
        self.n_pop = n_pop
        self.n_iter = n_iter    
        self.cv = cv      
        self.low_bound = low_bound
        self.upper_bound = upper_bound
        self.cross_proba = cross_proba
        
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
            self.alg_opt = _DifferentialEvolution(self.model,
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
                                      nPop = self.n_pop,
                                      low_bound = self.low_bound, 
                                      upper_bound = self.upper_bound, 
                                      cross_proba = self.cross_proba,
                                      verbose = self.verbose,
                                      random_state = self.random_state)     
            
        else:
            self.alg_opt = _DifferentialEvolution(self.model,
                                      self.cv,
                                      self.scorer,                                      
                                      self.n_f,
                                      self.fitness,
                                      self.weight,
                                      self.x,
                                      self.y,
                                      MaxIt = self.n_iter, 
                                      nPop = self.n_pop,
                                      low_bound = self.low_bound, 
                                      upper_bound = self.upper_bound, 
                                      cross_proba = self.cross_proba,
                                      verbose = self.verbose,
                                      random_state = self.random_state)     


        best_sol, self.best_fits, tops = self.alg_opt.generate()
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
        return "DE"
                
                
                
                
                
                
                
                
                
                
        