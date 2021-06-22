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
import copy
from itertools import chain

import time

class _GeneticAlgorithm():
    def __init__(self, estimator,cv, scorer, nf, fitness, weight,
                 x,y,x_test= None, y_test = None,
                 MaxIt = 1000, nPop = 10,  
                 cross_perc = 0.2, mut_perc = 0.3, mut_rate= 0.02, beta = 8,
                 random_state=None,verbose = True):
        
        self.nf = nf
        self.estimator = estimator
        self.cv = cv
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
        
        # ga parameters
        self.pc = cross_perc      #crossover percentage
        self.nc = int(self.pc*self.nPop) # number of offsprings
        
        self.pm = mut_perc   #mutation percentage
        self.nm = int(self.pm*self.nPop)
        
        self.mu = mut_rate  # mutation rate
        self.beta = beta   # selection pressure
        
        
        self.verbose = verbose
        self.random_state = random_state
        self.scorer = scorer        
        
        random.seed(self.random_state)        
        np.random.seed(self.random_state)                
        
    def RK(self,u):
        # this function creates random keys based on continuous values
        return np.argsort(u)[:self.nf]
        #return np.argsort(u)
        
    def LE(self,a,b):
        # this function returns the largest element between two arrays
        return np.array([max(i,b)for i in a])

    def SE(self,a,b):
        # this function returns the smallest element between two arrays
        return np.array([min(i,b) for i in a])        
        
    def Mutation(self,ind):
        # function that adds mutation to the initital position
        nmu = np.ceil(self.mu*self.VarSize).astype(int)
        index = random.sample(range(self.VarSize),nmu)
        
        sigma = 0.1*(self.VarMax - self.VarMin)
        muttd = copy.copy(ind)
        
        muttd[index] = ind[index] + sigma*np.random.normal(size=nmu)
        
        muttd = self.LE(muttd, self.VarMin)
        muttd = self.SE(muttd, self.VarMax)

        return muttd
    
    
    def CrossOver(self,ind1,ind2):
        # function that randomly chooses between three different cross over methods        
        # alpha = np.random.uniform(-self.gamma,1+self.gamma,self.VarSize)
        alpha = np.random.uniform(0,1,self.VarSize)

        
        ind3 = np.multiply(alpha,ind1) + np.multiply((1-alpha),ind2)
        ind4 = np.multiply(alpha,ind2) + np.multiply((1-alpha),ind1)
        
        ind3 = self.LE(ind3,self.VarMin)
        ind3 = self.SE(ind3,self.VarMax)

        ind4 = self.LE(ind4,self.VarMin)
        ind4 = self.SE(ind4,self.VarMax)
        
        return ind3,ind4
    
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

    # TODO expand cross over to three methods        
    # def UnifromCrossOver(ind1,ind2):        
        
    # def DoublePointCrossOver(ind1,ind2):
        
    # def SinglePointCrossOver(ind1,ind2):
        
    def RouletteWheelSelection(self,p):
        r = np.random.uniform(0, 1)
        c = np.cumsum(p)
        i = np.where(r <= c)[0][0]
        return i
    
    def generate(self):
        #instantate invdivual class
        Individual  = type('Individual', (object,), {})
        self.pop = [Individual() for i in range(self.nPop)]
        
        # instantiating best solution
        self.BestSol = Individual()
        if self.weight == 1:
            self.BestSol.fit = -np.inf
        elif self.weight == -1:
            self.BestSol.fit = np.inf
        
        # setting up a variable to store all agents 
        all_agents = []            
        
        # array to save best fits
        self.best_fits = np.zeros(self.MaxIt)          
        
        # initialize initial population and evaluate their cost function
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
        
        # survival of the fittest, initial population, the best individual is chosen
        self.fits = [individual.fit for individual in self.pop]
        sorted_ind = np.argsort(self.fits)
        self.pop = [self.pop[ind] for ind in sorted_ind] # sorting population based on best fit indices
        
        # store best individual in BestSol
        # if maximizing get the last individual in sorted pop
        if self.weight == 1:
            self.BestSol.position = self.pop[-1].position
            self.BestSol.fit = self.pop[-1].fit
            worst_fit = self.pop[0].fit #store worst fit            
        # if minimizing get the first individual in sorted pop
        elif self.weight == -1:
            self.BestSol.position = self.pop[0].position
            self.BestSol.fit = self.pop[0].fit
            worst_fit = self.pop[-1].fit #store worst fit
        
        #GA main loop
        for it in range(self.MaxIt):
            
            P = np.exp(-self.beta*np.array(self.fits)/worst_fit)
            P = P/sum(P)
            
            # cross over 
            popc = [[Individual(),Individual()] for i in range(self.nc)] #cross over population
            
            for k in range(self.nc):
                
                # parents indices
                par1_ind = self.RouletteWheelSelection(P)
                par2_ind = self.RouletteWheelSelection(P)
                
                # select parents
                par1 = self.pop[par1_ind]
                par2 = self.pop[par2_ind]
                
                # apply crossover
                offspring1, offspring2 = self.CrossOver(par1.position, par2.position)
                popc[k][0].position = offspring1
                popc[k][1].position = offspring2
                
                # evaluation
                if (self.x_test is not None) and (self.y_test is not None):
                    popc[k][0].fit, popc[k][0].perc = self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,self.RK(popc[k][0].position)],
                        self.y,
                        x_test=self.x_test[:,self.RK(popc[k][0].position)],
                        y_test=self.y_test)

                    popc[k][1].fit, popc[k][1].perc = self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,self.RK(popc[k][1].position)],
                        self.y,
                        x_test=self.x_test[:,self.RK(popc[k][1].position)],
                        y_test=self.y_test)
                    
                else:
                    popc[k][0].fit, popc[k][0].perc = self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,self.RK(popc[k][0].position)],
                        self.y)              
                
                    popc[k][1].fit, popc[k][1].perc = self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,self.RK(popc[k][1].position)],
                        self.y)                  
            
            # flattening the list of lists
            popc = [parent for parents in popc for parent in parents]
            
            # Apply mutation
            popm = [Individual() for i in range(int(self.nm))]  # mutation population
                  
            for k in range(self.nm):
                
                #select individual
                i = random.randint(0,self.nPop-1)
                par = self.pop[i]
                
                # apply mutation
                popm[k].position = self.Mutation(par.position)
                
                #Evaluate
                if (self.x_test is not None) and (self.y_test is not None):
                    popm[k].fit, popm[k].perc = self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,self.RK(popm[k].position)],
                        self.y,
                        x_test=self.x_test[:,self.RK(popm[k].position)],
                        y_test=self.y_test)   
                    
                else:
                    popm[k].fit, popm[k].perc = self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,self.RK(popm[k].position)],
                        self.y)                             
                
            # merge crossover and mutated population
            self.pop = list(chain(self.pop, popc, popm))
            
            # sort population based on fitness function
            self.fits = [individual.fit for individual in self.pop]
            sorted_ind = np.argsort(self.fits)
            self.pop = [self.pop[ind] for ind in sorted_ind]# sorting population based on best fit indices
            
            #update worst fit
            if self.weight == 1:
                worst_fit = min(worst_fit, self.pop[0].fit) 
                self.pop = self.pop[-self.nPop:]
                self.fits = self.fits[-self.nPop:]
            
            elif self.weight == -1:
                worst_fit = max(worst_fit, self.pop[-1].fit)            
                self.pop = self.pop[:self.nPop]
                self.fits = self.fits[:self.nPop]
                
            # store best solution found
            if self.weight == 1:
                self.BestSol.position = self.pop[-1].position
                self.BestSol.fit = self.pop[-1].fit
            elif self.weight == -1:
                self.BestSol.position = self.pop[0].position
                self.BestSol.fit = self.pop[0].fit
                
            self.best_fits[it] = self.BestSol.fit 
            
            # updating all agents
            all_agents.extend(self.pop)        
            
            if self.verbose:
                print("Iter: {} Features: {}  Fitness_score: {:.4} ".format(it+1,self.RK(self.BestSol.position),self.BestSol.fit))   
                
        # getting top_agents
        tops = self.FeatureRanker(all_agents) 
                
        return np.array([self.RK(self.BestSol.position), self.BestSol.fit]), self.best_fits, tops



class GA(BaseFeatureSelector):
    def __init__(self, model, n_f, weight, scoring, n_gen=1000, n_pop=20 , cv=None,                                
                 cross_perc = 0.5, mut_perc = 0.3, mut_rate= 0.02, beta = 5,
                 verbose= True, random_state=None,**kwargs):
        """
        Genetic Algorithms or GA is a widely used global optimization algorithm 
        which was first introduced by Holland. GA is based on the natural selection
        in the evolution theory. Properties of GA such as probability of mutation and 
        cross over determines the specifics of the search done in each iteration.
        Additionally, we can also set the proportion of the population we want to
        perform cross over or mutation for. 
                
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
        n_gen : int, optional
            Maximum number of generations or iterations. For more complex 
            problems it is better to set this parameter larger. 
            The default is 1000.
        n_pop : int, optional
            Number of population size at each iteration. Typically, this 
            parameter is set to be 10*n_f, but it is dependent on the complexity 
            of the model and it is advised that user tune this parameter based 
            on their problem. The default is 20.
        cv : class, optional
            Instantiated sklearn cross-validation class. The default is None.
        cross_perc : float, 
            The percentage of the population to perform cross over on. A common 
            choice for this parameter is 0.5. The larger cross_perc is chosen,
            the more exploition of the current population. The default is 0.5.
        mut_perc : float, optional
            The percentage of the population to perform mutation on. This is 
            usually chosen a small percentage (smaller than cross_perc). As 
            mut_perc is set larger, the model explorates more. 
            The default is 0.1.
        mut_rate : float, optional
            The mutation rate. This parameter determines the probability of 
            mutation for each individual in a population. It is often chosen 
            a small number to maintain the current good solutions.
            The default is 0.1.
        beta : int, optional
            Selection Pressure for cross-over. The higher this parameter the 
            stricter the selection of parents for cross-over. This value
            could be an integer [1,10]. The default value
            is 5.        
        verbose : bool, optional
            Wether to print out progress messages. The default is True.
        random_state : int, optional
            Determines the random state for random number generation, 
            cross-validation, and classifier/regression model setting. 
            The default is None.

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
        if n_pop != int(n_pop):
            raise ValueError("n_pop should be an integer")
        if n_gen != int(n_gen):
            raise ValueError("n_gen should be an integer")
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
        self.n_pop = n_pop
        self.n_gen = n_gen        
        
        self.cross_perc = cross_perc      #crossover percentage        
        self.mut_perc = mut_perc   #mutation percentage        
        self.mut_rate = mut_rate  # mutation rate
        self.beta = beta   # selection pressure
        # self.gamma = gamma        
        
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
            self.alg_opt = _GeneticAlgorithm(self.model,
                                      self.cv,
                                      self.scorer, 
                                      self.n_f,
                                      self.fitness,
                                      self.weight,
                                      self.x_train,
                                      self.y_train,                                                                             
                                      x_test = self.x_test,
                                      y_test = self.y_test,
                                      MaxIt = self.n_gen, 
                                      nPop = self.n_pop,
                                      cross_perc=self.cross_perc,
                                      mut_perc = self.mut_perc,
                                      mut_rate = self.mut_rate,
                                      beta = self.beta,
                                      verbose = self.verbose,
                                      random_state = self.random_state)     
            
        else:
            self.alg_opt = _GeneticAlgorithm(self.model,
                                      self.cv,
                                      self.scorer, 
                                      self.n_f,
                                      self.fitness,
                                      self.weight,
                                      self.x,
                                      self.y,
                                      MaxIt = self.n_gen, 
                                      nPop = self.n_pop,
                                      cross_perc=self.cross_perc,
                                      mut_perc = self.mut_perc,
                                      mut_rate = self.mut_rate,
                                      beta = self.beta,
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
        return "GA"                
                
                
                
                
                
                
                
                
                
        