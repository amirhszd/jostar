# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:24:31 2020

@author: Amirh
"""

from ..utils._base_feature_selector import BaseFeatureSelector
from ..functions import ff
import numpy as np
import random
from sklearn.base import clone
import time
from scipy.stats import iqr


class _AntColony():    
    def __init__(self, estimator,cv,scorer, nf, fitness, weight,
                 x,y,x_test= None, y_test = None,
                 MaxIt = 1000, n_ant = 10, Q=1, tau0=1, alpha=1, beta=1,
                 rho=0.5, random_state=None,verbose = True):
        
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
        self.n_ant = n_ant
        
        # ACO params
        self.Q = Q #pheromone strength
        self.tau0 = tau0 #initial phromone
        self.alpha = alpha #Phromone Exponential Weight
        self.beta = beta #Heuristic Exponential Weight
        self.rho = rho #Pheromone Evaporation rate
        self.n_var = self.x.shape[1]
        self.verbose = verbose
        self.random_state = random_state
        
        random.seed(self.random_state)        
        np.random.seed(self.random_state)
        
    def CreateRandomSolution(self):
        #Function producing random permutation
        return np.random.permutation(self.n_var)
        
    def RouletteWheelSelection(self,p):
        r = np.random.uniform(0, 1)
        c = np.cumsum(p)
        i = np.where(r <= c)[0][0]
        return i    
    
    def FeatureRanker(self,all_agents):
        agents = [agent.tour for agent in all_agents]
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
            
        eta = np.ones((self.n_var,self.n_var)) #heuristic information matrix
        tau = self.tau0*np.ones((self.n_var,self.n_var)) # phromone matrix
        
        self.best_fits = np.zeros(self.MaxIt)  # array holding best fits
        
        #creating the colony
        Ant = type("Ant", (object,), {})        
        colony = [Ant() for i in range(self.n_ant)]       
                
        # setting up a variable to store all agents 
        all_agents = []        
        
        # best ant 
        self.BestSol = Ant()
        
        if self.weight == 1:
            self.BestSol.fit = -np.inf
        elif self.weight == -1:
            self.BestSol.fit = np.inf
        
        for it in range(self.MaxIt):
            for k in range(self.n_ant):
                
                # you want to pay more attention to this part
                # initializing the first place to start at a random position
                colony[k].tour = [random.randint(0,self.n_var-1)]
                
                # building up on that intitial guess
                for l in np.arange(1,self.nf):
                    
                    i = colony[k].tour[-1]

                    P = np.power(np.power(np.power(tau[i,:],self.alpha),eta[i,:]),self.beta)
                    
                    
                    P[colony[k].tour] = 0
                    
                    P = P/sum(P)
                    
                    j = self.RouletteWheelSelection(P)
        
                    colony[k].tour = np.hstack([colony[k].tour,j])
                
                
                if (self.x_test is not None) and (self.y_test is not None):
                    colony[k].fit, colony[k].perc = self.fitness.calculate_fitness(self.estimator,
                          self.cv,
                          self.scorer,
                          self.x[:,colony[k].tour],
                          self.y,
                          x_test=self.x_test[:,colony[k].tour],
                          y_test=self.y_test)
                    
                else:
                    colony[k].fit, colony[k].perc = self.fitness.calculate_fitness(self.estimator,
                          self.cv,
                          self.scorer,
                          self.x[:,colony[k].tour],
                          self.y)
                
                # update the best ant
                if self.weight == 1 and colony[k].fit > self.BestSol.fit:
                    self.BestSol.tour = colony[k].tour
                    self.BestSol.fit = colony[k].fit
                    self.BestSol.perc = colony[k].perc
                    
                elif self.weight == -1 and colony[k].fit < self.BestSol.fit:
                    self.BestSol.tour = colony[k].tour
                    self.BestSol.fit = colony[k].fit
                    self.BestSol.perc = colony[k].perc
                    
                
            #update phromones
            for k in range(self.n_ant):
                
                tour = colony[k].tour
                tour = np.hstack([tour,tour[0]])
                
                for l in range(self.nf):
                    i = tour[l]
                    j = tour[l+1]
                    
                    tau[i,j] = tau[i,j] + self.Q/colony[k].fit
            
            # evaporate
            tau = (1-self.rho)*tau
            
            # best cost
            self.best_fits[it] = self.BestSol.fit
                        
            # updating all agents
            all_agents.extend(colony)
            
            if self.verbose:
                print("Iter: {} Features: {}  Fitness_score: {:.4} ".format(it+1, self.BestSol.tour[:self.nf],self.BestSol.fit))                        
        
        # getting top_agents
        tops = self.FeatureRanker(all_agents) 
        
        return np.array([self.BestSol.tour, self.BestSol.fit]), self.best_fits, tops
    

    
class ACO(BaseFeatureSelector):
    def __init__(self, model, n_f, weight, scoring, n_iter = 100, n_ant = 10, cv=None,
                 Q=1, tau0=1, alpha=1, beta=2, rho=0.5, 
                 verbose= True, random_state=None,**kwargs):
        """
        Ant Colony Optimization or ACO, is another nature inspired 
        optimization algorithm, which was introduced by M. Dorigo in 
        1992. In ACO, ants or agents of the model move around the 
        solution space randomly and remember the paths taken before 
        until they reach convergence. Flexibility of this algorithm
        enables us to employ it for feature selection.        

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
         as arguments
        n_iter : int, optional
            Maximum number of iterations. The default is 100.
        n_ant : int, optional
            Number of ants or agents in the optimzation model. The default is 10.
        cv : class, optional
            Instantiated sklearn cross-validation class. The default is None.
        Q : float, optional
            The pheromone intensity 𝑄, represents the total pheromonea vailable.
            This parameter affects the convergence speed of the ACO. A large 𝑄
            results in a highly concentrated pheromone, thus possibility of 
            converging to local optima. A small 𝑄 will results in possibility
            of attaining global optima, but slowers optimization speed. 
            The default is 1.
        tau0 : float, optional
            Initial pheromone intensity. This could be equal to Q. The default 
            is 1.
        alpha : float, optional
            The information elicitation factor 𝛼, addresses accumulation 
            of pheromone regarding path selection. 𝛼 should be [0,1] with an 
            emphasis on lower end of the range ([0,0.5]). If 𝛼 is large, the
             ants tend to choose the same path, chosen by preceding 
            ants, resulting in stronger cooperation among the ants and higher 
            convergence speed and higher possibility of falling into local 
            optimum. If 𝛼 is small,the convergence speed of the ACA is slowed 
            down, and ants can have a more global search to find optimum. 
            The default is 1.
        beta : float, optional
            The meta-heuristic factor 𝛽, reflects the importance of the 
            heuristic information with regard to the ants’ path selection. 𝛽
            is always larger than 1 and smaller than 5. If 𝛽 is large, there
            is a higher probability for the algorithm to behave like a 
            greedy algorithm. If 𝛽 is small, the heuristic information has no
             effect on path selection, which leads ACA to fall into local 
             optimum. The default is 2.
        rho : float, optional
            Pheromone Evaporation Coefficient 𝜌. Rrepresents the degree of 
            pheromone evaporation, reflecting the amount of mutual(social) 
            influence among ants. 𝜌 should be [0,1] with an emphasis on lower
            end of the range ([0,0.5]). If 𝜌 is small, the global search ability 
            will be reduced. If 𝜌 is too large, it will improve the global 
            search and higher chance of finding the global optimum. 
            The default is 0.5.
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
            Li, P., & Zhu, H. (2016). Parameter selection for ant colony 
            algorithm based on bacterial foraging algorithm. Mathematical 
            Problems in Engineering, 2016.

			Siemiński, A. (2013). Ant colony optimization parameter evaluation.
			In Multimedia and Internet Systems: Theory and Practice 
			(pp. 143-153). Springer, Berlin, Heidelberg.            

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
            raise ImportError("Cannot find/import ff.py defined")        

        if n_ant != int(n_ant):
            raise ValueError("n_ant should be an integer")
        if n_iter != int(n_iter):
            raise ValueError("n_iter should be an integer")

        self.model = model
        self.n_f = n_f
        self.cv = cv      
        self.weight = weight     
        self.scoring = scoring             
        self.n_ant = n_ant
        self.n_iter = n_iter  
        self.Q = Q
        self.tau0 = tau0 #initial phromone
        self.alpha = alpha #Phromone Exponential Weight
        self.beta = beta #Heuristic Exponential Weight
        self.rho = rho #Evaporation Rate   
        
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
            self.alg_opt = _AntColony(self.model,
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
                                      n_ant = self.n_ant,
                                      Q = self.Q,
                                      tau0 = self.tau0,
                                      alpha = self.alpha,
                                      beta = self.beta,
                                      rho = self.rho,
                                      verbose = self.verbose,
                                      random_state = self.random_state)     
            
        else:
            self.alg_opt = _AntColony(self.model,
                                      self.cv,
                                      self.scorer,
                                      self.n_f,
                                      self.fitness,
                                      self.weight,
                                      self.x,
                                      self.y,
                                      MaxIt = self.n_iter, 
                                      n_ant = self.n_ant,
                                      Q = self.Q,
                                      tau0 = self.tau0,
                                      alpha = self.alpha,
                                      beta = self.beta,
                                      rho = self.rho,
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
        return "ACO"
        
        
    
    

        
        
        