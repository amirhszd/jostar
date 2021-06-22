from ..utils._base_feature_selector import BaseFeatureSelector
from ..functions import ff
import numpy as np
from sklearn.base import is_regressor, clone
from itertools import permutations
import copy
import time
import random

class _ParticleSwarm():    
    def __init__(self, estimator,cv, scorer, nf, fitness, weight,
                 x,y,x_test= None, y_test = None,
                 MaxIt = 1000, nPop = 10, c1 = 0.5, c2 = 2, w = 0.5, wdamp = 0.9,
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
        
        # pso params
        self.VarSize = self.x.shape[1]
        self.VarMin = 0
        self.VarMax = 1

        self.w = w #inertia coefficinet
        self.wdamp = wdamp #intertia coefficient damping factor
        self.c1 = c1 # personal acceleration coefficient
        self.c2 = c2 # social acceleration coefficient
        
        self.VelMax = 0.1*(self.VarMax - self.VarMin)
        self.VelMin = - self.VelMax
        
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
        
        # Each particle has position, velocity and personal best 
        Particle = type("Particle", (object,), {})        
        swarm = [Particle() for i in range(self.nPop)]
        
        # best particle   
        self.BestSol = Particle()
        
        if self.weight == 1:
            self.BestSol.fit = -np.inf
        elif self.weight == -1:
            self.BestSol.fit = np.inf
    
        # array to save best fits
        self.best_fits = np.zeros(self.MaxIt)
        
        # setting up a variable to store all agents 
        all_agents = []                        
        
        for i in range(self.nPop):
            #initialize position
            swarm[i].position = np.random.uniform(self.VarMin,self.VarMax,self.VarSize)
            swarm[i].velocity = np.zeros(self.VarSize)
            
            # evaluation
            if (self.x_test is not None) and (self.y_test is not None):
                swarm[i].fit, swarm[i].perc = self.fitness.calculate_fitness(self.estimator,
                                  self.cv,
                                  self.scorer,
                                  self.x[:,self.RK(swarm[i].position)],
                                  self.y,
                                  x_test=self.x_test[:,self.RK(swarm[i].position)],
                                  y_test=self.y_test)
                
            else:
                swarm[i].fit, swarm[i].perc = self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,self.RK(swarm[i].position)],
                        self.y)
              
            # update personal best
            swarm[i].best_position = swarm[i].position
            swarm[i].best_fit = swarm[i].fit
        
            #update global best
            if self.weight == 1 and swarm[i].best_fit > self.BestSol.fit:
                self.BestSol.position = swarm[i].best_position
                self.BestSol.fit = swarm[i].best_fit
                
            elif self.weight == -1 and swarm[i].fit < self.BestSol.fit:
                self.BestSol.position = swarm[i].best_position
                self.BestSol.fit = swarm[i].best_fit
         
        # main loop
        for it in range(self.MaxIt):
            for i in range(self.nPop):
                
                # we start updating new particle positions by calculating new velocity
                # wich is a compromise between Inertial term, cognitive and social components
                
                # update velocity
                swarm[i].velocity = (self.w*swarm[i].velocity + # inertia term
                                     np.power(self.c1*np.random.uniform(self.VarMin, self.VarMax, self.VarSize),swarm[i].best_position - swarm[i].position) + #cognitive component
                                     np.power(self.c2*np.random.uniform(self.VarMin, self.VarMax, self.VarSize),self.BestSol.position- swarm[i].position)) #social component

                # apply velocity limits
                swarm[i].velocity = self.LE(swarm[i].velocity,self.VelMin)
                swarm[i].velocity = self.SE(swarm[i].velocity,self.VelMax)
                        
                # updating position based on new velocity vector
                swarm[i].position = swarm[i].position + swarm[i].velocity
                
                # velocity mirror effect
                isoutside = (swarm[i].position< self.VarMin) | (swarm[i].position> self.VarMax)
                changed_values = copy.copy(swarm[i].velocity)
                changed_values[isoutside] = -changed_values[isoutside]                
                swarm[i].velocity=changed_values

                # apply position limites
                swarm[i].position = self.LE(swarm[i].position,self.VarMin)
                swarm[i].position = self.SE(swarm[i].position,self.VarMax)
        
                # evaluation
                if (self.x_test is not None) and (self.y_test is not None):
                    swarm[i].fit, swarm[i].perc = self.fitness.calculate_fitness(self.estimator,
                            self.cv,
                            self.scorer,
                            self.x[:,self.RK(swarm[i].position)],
                            self.y,
                            x_test=self.x_test[:,self.RK(swarm[i].position)],
                            y_test=self.y_test)
                    
                else:
                    swarm[i].fit, swarm[i].perc = self.fitness.calculate_fitness(self.estimator,
                            self.cv,
                            self.scorer,
                            self.x[:,self.RK(swarm[i].position)],
                            self.y) 
        
                # update personal best
                if self.weight == 1 and swarm[i].fit > swarm[i].best_fit:
                    swarm[i].best_position = swarm[i].position
                    swarm[i].best_fit = swarm[i].fit
                    
                    #update global best
                    if swarm[i].best_fit > self.BestSol.fit:
                        self.BestSol.position = swarm[i].best_position
                        self.BestSol.fit = swarm[i].best_fit
                    
                elif self.weight == -1 and swarm[i].fit < swarm[i].best_fit:
                    swarm[i].best_position = swarm[i].position
                    swarm[i].best_fit = swarm[i].fit
                    
                    #update global best
                    if swarm[i].best_fit < self.BestSol.fit:
                        self.BestSol.position = swarm[i].best_position
                        self.BestSol.fit = swarm[i].best_fit
        
            self.best_fits[it] = self.BestSol.fit 
            
            # updating all agents
            all_agents.extend(swarm)       
            
            if self.verbose:
                print("Iter: {} Features: {}  Fitness_score: {:.4} ".format(it+1,self.RK(self.BestSol.position),self.BestSol.fit))                        
        
            self.w = self.w*self.wdamp
            
        # getting top_agents
        tops = self.FeatureRanker(all_agents) 
            
        return np.array([self.RK(self.BestSol.position), self.BestSol.fit]), self.best_fits, tops
    

                 
class PSO(BaseFeatureSelector):
    def __init__(self, model, n_f, weight, scoring, n_iter=1000, n_pop=20, cv=None, 
                 c1 = 2, c2 = 2, w = 1.2,wdamp = 0.25,
                 verbose= True, random_state=None,**kwargs):       
                
        """
        PSO or Particle Swarm Optimization was first introduced by J. Kennedy 
        and R. Eberhart in 1995. This algorithm mimics the behavior of the 
        bird flocks in the nature.
        
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
            Number of population size at each iteration. Typically, this 
            parameter is set to be 10*n_f, but it is dependent on the complexity 
            of the model and it is advised that user tune this parameter based 
            on their problem. The default is 20.
        cv : class, optional
            Instantiated sklearn cross-validation class. The default is None.
        c1 : float, optional
            The cognitive parameter. This parameter determines the coefficient 
            of particl's next location. The higher this coefficient the particle
            tends to move toward the personal best. c1 + c2 <=4 and also need 
            to be fine-tuned. The default is 2.
        c2 : float, optional
            The social parameter. This parameter determines the coefficient 
            of particl's next location. The higher this coefficient the particle
            tends to move toward the global best. c1 + c2 <=4 and also need 
            to be fine-tuned. The default is 2.
        w : float, optional
            Inertia weight is critical for convergence behaviour. w is a trade–off between 
            the global and local exploration of the swarm. A large inertia weight 
            explore more while a smaller one tends to focus on local exploration.
            Usually a value is often chosen between [0,1.2]. The default value is 1.2.
        wdamp : float, optional
            Scaling factor or damping factor for intertia weight. This value
            is often chosen between [0,0.5] and always below 1. 
            The default is 0.25.                  
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
            Beielstein, T., Parsopoulos, K. E., & Vrahatis, M. N. (2002). 
            Tuning PSO parameters through sensitivity analysis.
            Universitätsbibliothek Dortmund.            

        """            
        
        super().__init__(scoring=scoring, n_f = n_f,**kwargs)        
# =============================================================================
# checking the input arguments
# =============================================================================
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
        self.cv = cv      
        self.weight = weight     
        self.n_pop = n_pop
        self.n_iter = n_iter    
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.wdamp = wdamp
        
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
            self.alg_opt = _ParticleSwarm(self.model,
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
                                      c1= self.c1, 
                                      c2 = self.c2, 
                                      w= self.w,
                                      wdamp = self.wdamp,
                                      verbose = self.verbose,
                                      random_state = self.random_state)                                          
            
        else:
            self.alg_opt = _ParticleSwarm(self.model,
                                      self.cv,
                                      self.scorer,                                      
                                      self.n_f,
                                      self.fitness,
                                      self.weight,
                                      self.x,
                                      self.y,                                                                             
                                      MaxIt = self.n_iter, 
                                      nPop = self.n_pop,
                                      c1= self.c1, 
                                      c2 = self.c2, 
                                      w= self.w,
                                      wdamp = self.wdamp,
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
        return "PSO"        
        
        
        
        
        
        
        
    
    