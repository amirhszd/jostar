# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:58:32 2020

@author: Amirh
"""
from ..utils._base_feature_selector import BaseFeatureSelector
from ..functions import ff_nsga2
from ..metrics._metrics import adj_r2_score, rmspe_score,cross_validate_LOO
import numpy as np
from sklearn.base import clone, is_regressor
import random
import copy
import time
import matplotlib.pyplot as plt
import warnings
from itertools import chain
import pandas as pd
from ast import literal_eval
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_predict
import seaborn as sns
from sklearn.metrics import *

class _NonDominatedSortingGA():
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
        self.nObj = len(self.weight)
        
        # nsga2 parameters
        self.pc = cross_perc      #crossover percentage
        self.nc = int(self.pc*self.nPop) # number of offsprings
        
        self.pm = mut_perc   #mutation percentage
        self.nm = int(self.pm*self.nPop)
        
        self.mu = mut_rate  # mutation rate

        self.verbose = verbose
        self.random_state = random_state
        self.scorer = scorer        
        
        random.seed(self.random_state)        
        np.random.seed(self.random_state)    
        
    
    def Dominates(self,x,y):
        x = np.array(x.fit)
        y = np.array(y.fit)
        
        # first we need all of them to be at leشst eq smaller/higher than the other one
        bees_all = []
        for w in range(len(self.weight)):
            if self.weight[w]==1: #maximizing
                b = x[w] >= y[w]
            elif self.weight[w] == -1: #minimizing
                b = x[w] <= y[w]            
            bees_all.append(b)
        
        bees_all = np.array(bees_all).all() 
        # if this was true check to see if there is at least one that satisfies
        if bees_all == True:
            bees_any = [] 
            for w in range(len(self.weight)):
                if self.weight[w]==1: #maximizing
                    b = x[w] > y[w]
                elif self.weight[w] == -1: #minimizing
                    b = x[w] < y[w]
                bees_any.append(b)
                
            bees_any = np.array(bees_any).any()
            return bees_any
        else:
            return bees_all
           
        
    def NonDominatedSorting(self):
        self.F = []

        for i in range(len(self.pop)):
            self.pop[i].domination_set = []
            self.pop[i].dominated_count = 0            
        
        for i in range(len(self.pop)):            
            for j in range(i+1,len(self.pop)):
                
                # if p dominates q
                if self.Dominates(self.pop[i],self.pop[j]):
                    self.pop[i].domination_set.append(j)
                    self.pop[j].dominated_count += 1
                    
                # if q dominates p
                if self.Dominates(self.pop[j],self.pop[i]):
                    self.pop[j].domination_set.append(i)
                    self.pop[i].dominated_count += 1
                    
            # if domination count is zero pass it as a front
            if self.pop[i].dominated_count == 0:
                self.F.append(i)
                self.pop[i].rank = 1
                
        k = 0
        self.F = [self.F]
        while True:
            Q = []
            for i in self.F[k]:   
                for j in self.pop[i].domination_set:
                    self.pop[j].dominated_count += -1
                    
                    if self.pop[j].dominated_count == 0:
                        Q.append(j)
                        self.pop[j].rank = k+2
                                     
            if len(Q) == 0:
                break
            
            self.F.append(Q)
            k += 1

    def NonDominatedSorting_fronts(self,old_fronts, new_fronts):     
        updated_fronts = copy.copy(old_fronts)
        for i in range(len(new_fronts)):    
            # if new_front dominates old_front
            for j in range(len(old_fronts)):
                if new_fronts[i].nf == old_fronts[j].nf:
                    if self.Dominates(new_fronts[i],old_fronts[j]):
                        updated_fronts.remove(old_fronts[j])
                        updated_fronts.append(new_fronts[i])
                    break
                    
        return updated_fronts
    
    def CalcCrowdingDistance(self):
        nF = len(self.F)
        
        for k in range(nF):
            fits = np.array([self.pop[index].fit for index in self.F[k]]) # individuals x number of features
            n = len(self.F[k])
            
            d = np.zeros((n,self.nObj)) # distance
            
            for j in range(self.nObj):
                
                if fits.shape[0] != 1:
                    ind = np.argsort(fits[:,j])
                    cj = fits[ind,j]
                
                    d[ind[0],j] = np.inf
                    for i in range(1,n-1):
                        d[ind[i],j] = abs(cj[i+1] - cj[i-1])/abs(cj[0] - cj[-1])
                    d[ind[-1],j] = np.inf
                
                else:
                    ind = 0
                    d[0,j] = np.inf
            
            for i in range(n):
                self.pop[self.F[k][i]].crowding_dist = sum(d[i,:])
                                
                
    def SortPopulation(self):
        #sort based on crowding distance
        CDSO = np.argsort([i.crowding_dist for i in self.pop])[::-1]
        self.pop = [self.pop[ind] for ind in CDSO]
        
        # sort based on rank
        RSO = np.argsort([i.rank for i in self.pop])
        self.pop = [self.pop[ind] for ind in RSO ]
        
        # update fronts
        ranks = [i.rank for i in self.pop]
        maxrank = max(ranks)
        self.F = [[] for i in range(maxrank)]
        
        for r in range(1,maxrank+1):
            self.F[r-1] = np.where(np.array(ranks) ==r)[0]
                        
                            
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
        # function that applies crossover
        alpha = np.random.uniform(size = self.VarSize)
        
        ind3 = np.multiply(alpha,ind1) + np.multiply((1-alpha),ind2)
        ind4 = np.multiply(alpha,ind2) + np.multiply((1-alpha),ind1)
        
        ind3 = self.LE(ind3,self.VarMin)
        ind3 = self.SE(ind3,self.VarMax)

        ind4 = self.LE(ind4,self.VarMin)
        ind4 = self.SE(ind4,self.VarMax)
        
        return ind3,ind4   
    
    def Plotfits(self,Fronts):
                
        if len(self.weight) >= 2 :
            n_row = np.ceil(len(self.weight)/2).astype(int)
            n_col = 2
                        
            fits = np.array([front.fit for front in Fronts])            
            nfs = np.array([front.nf for front in Fronts])  
            for n in range(len(self.weight)):        
                axes[n].clear()                
                axes[n].scatter(nfs, fits[:,n], color='r')        
                axes[n].set_ylabel("Objective {}".format(n+1))
                axes[n].set_xlabel("Number of features")    
                axes[n].grid(True)
            plt.tight_layout()
            plt.show(block=False)     
            plt.pause(1)            
        else:
            pass
        
        
    def RemoveFalseMembers(self):
        remove_set= set()
        for i in range(len(self.pop)):
            if type(self.pop[i].crowding_dist) == list:
                remove_set.add(i)
            if type(self.pop[i].rank) == list:
                remove_set.add(i)
                
        keep_list =  [i for i in range(len(self.pop)) if i not in remove_set]
        self.pop = [self.pop[ind] for ind in keep_list]

        
    def RemoveDuplicate(self,Fronts):
        # remove duplicates based on positions
        Fronts_pos = [list(self.RK_nf(front.position,front.nf)) for front in Fronts]
        # sort Front_pos based on number of features
        sorted_Fronts_pos = sorted(Fronts_pos, key=len)
        soreted_Fronts_pos_ind = [Fronts_pos.index(i) for i in sorted_Fronts_pos]
        Fronts = [Fronts[ind] for ind in soreted_Fronts_pos_ind]        
        
        # getting unique values                        
        _, indices = np.unique(sorted_Fronts_pos,True)        
        indices = np.sort(indices)                        
        return [Fronts[ind] for ind in indices]        
    
    def RK(self,u):
        # this function creates random keys based on continuous values
        indices = np.random.randint(1,self.nf+1)
        return np.argsort(u)[:indices]

    def RK_nf(self,u,nf):
        # return np.argsort(u)[:nf]
        return sorted(np.argsort(u)[:nf])
    
    def LE(self,a,b):
        # this function returns the largest element between two arrays
        return np.array([max(i,b) for i in a])

    def SE(self,a,b):
        # this function returns the smallest element between two arrays
        return np.array([min(i,b) for i in a])            
    
    def M(self,x):
        #return np.multiply(np.multiply(self.weight, -1),x) # NSGA originally computes minimization
        return x
            
    def generate(self):
        global Individual, fig, axes
        plt.ion()
        #instantate invdivual class
        Individual  = type('Individual', (object,), {'crowding_dist':[],'rank' : []})
        self.pop = [Individual() for i in range(self.nPop)]        


        # initialize initial population and evaluate their cost function
        for i in range(self.nPop):           
            self.pop[i].position = np.random.uniform(self.VarMin,self.VarMax,self.VarSize)  
            # get a random number of features from it
            indices = self.RK(self.pop[i].position)
            self.pop[i].nf = len(indices)
            # evaluation
            if (self.x_test is not None) and (self.y_test is not None):
                self.pop[i].fit = self.M(self.fitness.calculate_fitness(self.estimator,self.cv,self.scorer,
                                                                        self.x[:,indices],
                                                                        self.y,
                                                                        x_test=self.x_test[:,indices],
                                                                        y_test=self.y_test))  

                
            else:
                self.pop[i].fit = self.M(self.fitness.calculate_fitness(self.estimator,
                                                                        self.cv,
                                                                        self.scorer,
                                                                        self.x[:,indices],
                                                                        self.y))
                
        # Non_diminated sorting
        self.NonDominatedSorting()
        
        # calcualate crowding distance
        self.CalcCrowdingDistance()
                
        # sort population
        self.SortPopulation()        
        
        # Open a new figure
        fig , axes = plt.subplots(1,len(self.weight),
                                  figsize=(10 + (len(self.weight) - 2)*5,5 + (len(self.weight) - 2)*5),
                                  )      
        
        #NSGA2 main loop
        for it in range(self.MaxIt):
            
            # crossover
            popc = [[Individual(),Individual()] for i in range(self.nc)] #cross over population
            for k in range(self.nc):
                
                # choosing two parents
                par1_ind = np.random.randint(0,self.nPop)
                par2_ind = np.random.randint(0,self.nPop)

                par1 = self.pop[par1_ind] 
                par2 = self.pop[par2_ind]
                
                # cross over parents to make new offsprings
                offspring1, offspring2 = self.CrossOver(par1.position, par2.position)

                popc[k][0].position = offspring1
                popc[k][1].position = offspring2
                
                # grabbing random key for each offspring
                indices1 = self.RK(popc[k][0].position)
                popc[k][0].nf = len(indices1)
                indices2 = self.RK(popc[k][1].position)
                popc[k][1].nf = len(indices2)
                
                # evaluation
                if (self.x_test is not None) and (self.y_test is not None):
                    popc[k][0].fit = self.M(self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,indices1],
                        self.y,
                        x_test=self.x_test[:,indices1],
                        y_test=self.y_test))
                    
                    popc[k][1].fit = self.M(self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,indices2],
                        self.y,
                        x_test=self.x_test[:,indices2],
                        y_test=self.y_test))                 
                    
                else:   
                    
                    popc[k][0].fit = self.M(self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,indices1],
                        self.y))                
                
                    popc[k][1].fit = self.M(self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,indices2],
                        self.y))
                
            popc = [parent for parents in popc for parent in parents]
            
            # MUTATION
            popm = [Individual() for i in range(int(self.nm))]  # mutation population
                  
            for k in range(self.nm):
                
                #select individual
                i = random.randint(0,self.nPop-1)
                par = self.pop[i]
                
                # apply mutation
                popm[k].position = self.Mutation(par.position)
                
                #Evaluate
                indices = self.RK(popm[k].position)
                popm[k].nf = len(indices)
                if (self.x_test is not None) and (self.y_test is not None):

                    popm[k].fit = self.M(self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,indices],
                        self.y,
                        x_test=self.x_test[:,indices],
                        y_test=self.y_test))
                    
                else:
                    
                    popm[k].fit = self.M(self.fitness.calculate_fitness(self.estimator,
                        self.cv,
                        self.scorer,
                        self.x[:,indices],
                        self.y))                  
                    
            #MERGE SETS
            self.pop = list(chain(self.pop, popc, popm))
            
            # NON DOMINATED SORTING
            self.NonDominatedSorting()

            #CALCULATE CROWDING DISTANCE
            self.CalcCrowdingDistance()

            # REMOVE MEMBERS WITH NO RANK/CROWDING DISTANCE
            #self.RemoveFalseMembers()
                                            
            # SORT POPULATION
            self.SortPopulation()
            
            # TRUNCATE        
            self.pop = self.pop[:self.nPop]
            
            # NON DOMINATED SORTING
            self.NonDominatedSorting()
            
            #CALCULATE CROWDING DISTANCE
            self.CalcCrowdingDistance()                            

            # REMOVE MEMBERS WITH NO RANK/CROWDING DISTANCE
            #self.RemoveFalseMembers()
            
            # SORT POPULATION
            self.SortPopulation()
            
            # STORE F0
            new_Fronts = np.array(self.pop)[self.F[0]]            
            
            # show pareto front
            new_Fronts = self.RemoveDuplicate(new_Fronts)

            # see if new_fronts dominate old_fronts            
            if it > 0:
            	#sometimes it doesnt find anything!
                if len(new_Fronts) > 0:
                    new_Fronts = self.NonDominatedSorting_fronts(old_Fronts,new_Fronts) # OK
                    new_Fronts = self.RemoveDuplicate(new_Fronts)           
            
            if self.verbose:
                print("Iteration: {} completed with {} fronts".format(it+1,len(new_Fronts)))

            if len(new_Fronts) > 0:                
                self.Plotfits(new_Fronts)
            old_Fronts = copy.copy(new_Fronts) # OK

        self.pareto_front = (fig,axes)
        return [[self.RK_nf(front.position,front.nf), front.fit] for front in new_Fronts]


class NSGA2(BaseFeatureSelector):
    def __init__(self, model, n_f, weight, scoring, n_gen=1000, n_pop=20, cv=None,
                 cross_perc = 0.5, mut_perc = 0.3, mut_rate= 0.02, beta = 5,
                 verbose= True, random_state=None,**kwargs):
        """
        Non-Dominated Sorting Genetic Algorithms 2 or NSGA2 is fast multi-objective
        optimization algorihtm introduced by K. Deb. et al. (2000). This algorithm
        takes advantage of the widely-known GA and uses dominated sorting to 
        best solution in multi-deimenstional/multi-objective search space. GA 
        is based on the natural selection in the evolution theory. Properties of GA such as probability of mutation and 
        cross over determines the specifics of the search done in each iteration.
        Additionally, we can also set the proportion of the population we want to
        perform cross over or mutation for. 
                
        Parameters
        ----------
        model : class
            Instantiated Sklearn regression or classification estimator.
        n_f : int
            Number of features needed to be extracted.
        weights : tuple
            A tuple indicating maximization or minimization objective. 
            For example for a bi-objective task of max and min function the 
            weights should be (+1,-1).
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
        if not all(i == int(i) for i in weight):
            raise ValueError("weight should be a tuple of either -1 or +1")
        try:
            self.fitness = ff_nsga2
        except:
            raise ImportError("Cannot find/import ff.py defined")
        
        self.model = model
        self.n_f = n_f
        self.cv = cv      
        self.weight = weight     
        self.n_pop = n_pop
        self.n_gen = n_gen        
        self.cross_perc = cross_perc
        self.mut_perc = mut_perc
        self.mut_rate = mut_rate
        self.beta = beta
        
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
        best_sols: List of length n_f
            Best solution for each number of feature prior to preprocessing.
        best_sols_acc:  List of length n_f
            Fitness value for each number of feature.
        best_sols_decor: List of length n_f
            Best solution for each number of feature after preprocessing.
            If no decor value is given, best_sols_decor equals best_sols.
        model_best: Class
            Regression/classification estimator given to the optimization 
            algorithm. At this stage model_best == model

        Returns
        -------
        Dataframe 
            Dataframe with number of features selected and their corresponding 
            indices and accuracy metrics. 
        """
        self.load(x,y,decor=decor,scale=scale)        
        if test_size is not None:
            self.train_test_split(test_size, random_state=self.random_state,**kwargs)

        
        a = time.time()
        if hasattr(self, 'x_train') and hasattr(self, 'x_test') and hasattr(self, 'y_train') and hasattr(self, 'y_test'):
            self.alg_opt = _NonDominatedSortingGA(self.model,
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
                                      cross_perc = self.cross_perc,
                                      mut_perc = self.mut_perc,
                                      mut_rate = self.mut_rate,
                                      beta = self.beta,
                                      verbose = self.verbose,
                                      random_state = self.random_state)     
            
        else:
            self.alg_opt = _NonDominatedSortingGA(self.model,
                                      self.cv,
                                      self.scorer, 
                                      self.n_f,
                                      self.fitness,
                                      self.weight,
                                      self.x,
                                      self.y,                                                                             
                                      MaxIt = self.n_gen, 
                                      nPop = self.n_pop,
                                      cross_perc = self.cross_perc,
                                      mut_perc = self.mut_perc,
                                      mut_rate = self.mut_rate,
                                      beta = self.beta,
                                      verbose = self.verbose,
                                      random_state = self.random_state)     

        best_sols = self.alg_opt.generate()
        self.best_sols = [str(self.x_cols[i[0]].to_list()) for i in best_sols] # positions
        self.best_sols_decor = [str(i[0]) for i in best_sols] # positions
        self.best_sols_acc = np.array([i[1] for i in best_sols]) #fits
        
        self.pareto_front = self.alg_opt.pareto_front
        
        # setting the model_best = model since one may not want to run tune_param
        self.model_best = clone(self.model)
        if self.verbose:
            print("Optimization completed in {} seconds".format(time.time() - a))
        
        # showing the data as dataframe
        res_df = pd.DataFrame()
        res_df["Feauteres_decor_ind"] = self.best_sols_decor
        res_df["Feauteres_org_ind"] = self.best_sols
        for w in range(len(self.weight)):
            res_df["Obj_{}".format(w+1)] = self.best_sols_acc[:,w]                

        self.res_df = res_df                        
        return self.res_df
    
    # defining a new display_results function
    
    def display_results(self,index):
        """
        this function generates regression plots of calibration and 
        cross-validation sets given an index of the generated dataframe.        

        Parameters
        ----------
        index : int
            Index (row index) of the subset ouputted by generated dataframe.

        Returns
        -------
        None.

        """
        self.best_features = literal_eval(self.res_df.iloc[index]["Feauteres_decor_ind"])
                
        if self.cv is not None:
            # if task is regression
            if is_regressor(self.model):
# =============================================================================
#           Running with general algorithm and calculating accuracies
# =============================================================================                

                # calculating accuracies for generalized model             
                self.model.fit(self.x,self.y)
                self.y_pred_cal_gen = np.squeeze(self.model.predict(self.x)).squeeze()
                self.r2_cal_gen = r2_score(self.y,self.y_pred_cal_gen)
                self.r2_adj_cal_gen = adj_r2_score(self.y,self.y_pred_cal_gen, self.x, self.r2_cal_gen)
                self.rmse_cal_gen = mean_squared_error(self.y, self.y_pred_cal_gen, squared = False)
                self.rmspe_cal_gen = rmspe_score(self.y, self.y_pred_cal_gen)

                if self.cv.__str__() == "LeaveOneOut()":
                    score_r2 = cross_validate_LOO(self.model,self.x,self.y,self.cv,
                                                  scoring=r2_score)                    

                    score_rmse = cross_validate_LOO(self.model,self.x,self.y,self.cv,
                                                    scoring=mean_squared_error)
                    
                    self.y_pred_cv_gen = cross_val_predict(self.model, self.x, self.y, n_jobs = -1, cv=self.cv).squeeze()        
                    self.r2_cv_gen = score_r2          
                    self.r2_adj_cv_gen = adj_r2_score(self.y, x=self.x, r2=self.r2_cv_gen)
                    self.rmse_cv_gen = np.sqrt(score_rmse)
                    self.rmspe_cv_gen = rmspe_score(self.y, self.y_pred_cv_gen)                    
                    
                else:
                    scores = cross_validate(self.model,self.x,self.y,
                                                  n_jobs = -1, 
                                                  cv=self.cv,
                                                  scoring=['r2','neg_mean_squared_error'])    
                    
                    self.y_pred_cv_gen = cross_val_predict(self.model, self.x, self.y, n_jobs = -1, cv=self.cv).squeeze()        
                    self.r2_cv_gen = scores['test_r2'].mean()            
                    self.r2_adj_cv_gen = adj_r2_score(self.y, x=self.x, r2=self.r2_cv_gen)
                    self.rmse_cv_gen = np.sqrt(scores['test_neg_mean_squared_error'].mean())
                    self.rmspe_cv_gen = rmspe_score(self.y, self.y_pred_cv_gen)                                             

                
    # =============================================================================
    #           Running optimized model and calculate accuracies
    # =============================================================================
                    
                self.model_best.fit(self.x[:,self.best_features],self.y)            
                self.y_pred_cal_op = np.squeeze(self.model_best.predict(self.x[:,self.best_features])).squeeze()
                self.r2_cal_op = r2_score(self.y,self.y_pred_cal_op)
                self.r2_adj_cal_op = adj_r2_score(self.y,self.y_pred_cal_op,self.x[:,self.best_features],self.r2_cal_op)
                self.rmse_cal_op = mean_squared_error(self.y, self.y_pred_cal_op, squared = False)
                self.rmspe_cal_op = rmspe_score(self.y, self.y_pred_cal_op)
                                
                if self.cv.__str__() == "LeaveOneOut()":
                    
                    score_r2 = cross_validate_LOO(self.model_best,self.x[:,self.best_features],self.y,self.cv,
                                                  scoring=r2_score)                    

                    score_rmse = cross_validate_LOO(self.model_best,self.x[:,self.best_features],self.y,self.cv,
                                                    scoring=mean_squared_error)                    
                    
                    self.y_pred_cv_op = cross_val_predict(self.model_best, self.x[:,self.best_features], self.y, n_jobs = -1, cv=self.cv).squeeze()                                    
                    self.r2_cv_op = score_r2   
                    self.r2_adj_cv_op = adj_r2_score(self.y,x = self.x[:,self.best_features],r2= self.r2_cv_op)
                    self.rmse_cv_op = np.sqrt(score_rmse)
                    self.rmspe_cv_op = rmspe_score(self.y, self.y_pred_cv_op)                    
                    

                else:
                    scores = cross_validate(self.model_best,self.x[:,self.best_features],self.y,
                                            n_jobs = -1, 
                                            cv=self.cv,
                                            scoring=['r2','neg_mean_squared_error'])                    
                    
                    self.y_pred_cv_op = cross_val_predict(self.model_best, self.x[:,self.best_features], self.y, n_jobs = -1, cv=self.cv).squeeze()                                    
                    self.r2_cv_op = scores['test_r2'].mean()     
                    self.r2_adj_cv_op = adj_r2_score(self.y,x = self.x[:,self.best_features],r2= self.r2_cv_op)
                    self.rmse_cv_op = np.sqrt(scores['test_neg_mean_squared_error'].mean())
                    self.rmspe_cv_op = rmspe_score(self.y, self.y_pred_cv_op)

    # =============================================================================
    #           Plotting generalized model and optimized model        
    # =============================================================================
                # generalized model
                fig,axes = plt.subplots(2,3,figsize=(15,10))
                sns.regplot(x=self.y, y=self.y_pred_cal_gen, color="k",ax=axes[0, 0])
                axes[0,0].plot([np.min([self.y, self.y_pred_cal_gen]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cal_gen]) + 0.1*np.std(self.y)],
                    [np.min([self.y, self.y_pred_cal_gen]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cal_gen]) + 0.1*np.std(self.y)],'r--')
                axes[0,0].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_cal_gen) + "\n"+ 
                                       r"RMSPE={:.3f}".format(self.rmspe_cal_gen) + "\n"+ 
                                       r"$R^2$={:.3f}".format(self.r2_cal_gen) + "\n"+
                                       r"$Adj. R^2$={:.3f}".format(self.r2_adj_cal_gen),
                                       transform=axes[0, 0].transAxes,
                                       color='k',
                                       bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                axes[0,0].set_xlabel("Measured")
                axes[0,0].set_ylabel("Predicted")
                axes[0,0].set_title("General model - Cal.")
                axes[0,0].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))
                
                sns.regplot(x=self.y, y=self.y_pred_cv_gen, color="k",ax=axes[1, 0])
                axes[1,0].plot([np.min([self.y, self.y_pred_cv_gen]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cv_gen]) + 0.1*np.std(self.y)],
                    [np.min([self.y, self.y_pred_cv_gen]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cv_gen]) + 0.1*np.std(self.y)],'r--')            
                axes[1,0].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_cv_gen) + "\n"+ 
                                       r"RMSPE={:.3f}".format(self.rmspe_cv_gen) + "\n"+ 
                                       r"$R^2$={:.3f}".format(self.r2_cv_gen) + "\n"+
                                       r"$Adj. R^2$={:.3f}".format(self.r2_adj_cv_gen),
                                       transform=axes[1, 0].transAxes,
                                       color='k',
                                       bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                axes[1,0].set_xlabel("Measured")
                axes[1,0].set_ylabel("Predicted")
                axes[1,0].set_title("General model - CV")
                axes[1,0].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))
    
                # optimized model
                sns.regplot(x=self.y, y=self.y_pred_cal_op, color="k",ax=axes[0, 1])
                axes[0,1].plot([np.min([self.y, self.y_pred_cal_op]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cal_op]) + 0.1*np.std(self.y)],
                    [np.min([self.y, self.y_pred_cal_op]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cal_op]) + 0.1*np.std(self.y)],'r--')               
                axes[0,1].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_cal_op) + "\n"+ 
                                       r"RMSPE={:.3f}".format(self.rmspe_cal_op) + "\n"+ 
                                       r"$R^2$={:.3f}".format(self.r2_cal_op)+ "\n"+
                                       r"$Adj. R^2$={:.3f}".format(self.r2_adj_cal_op),
                                       transform=axes[0, 1].transAxes,
                                       color='k',
                                       bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                axes[0,1].set_xlabel("Measured")
                axes[0,1].set_ylabel("Predicted")
                axes[0,1].set_title("Optimized model - Cal.")
                axes[0,1].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                   
                
                sns.regplot(x=self.y, y=self.y_pred_cv_op, color="k",ax=axes[1, 1])
                axes[1,1].plot([np.min([self.y, self.y_pred_cv_op]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cv_op]) + 0.1*np.std(self.y)],
                    [np.min([self.y, self.y_pred_cv_op]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cv_op]) + 0.1*np.std(self.y)],'r--')              
                axes[1,1].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_cv_op) + "\n"+ 
                                       r"RMSPE={:.3f}".format(self.rmspe_cv_op) + "\n"+ 
                                       r"$R^2$={:.3f}".format(self.r2_cv_op)+ "\n"+
                                       r"$Adj. R^2$={:.3f}".format(self.r2_adj_cv_op),
                                       transform=axes[1, 1].transAxes,
                                       color='k',
                                       bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                axes[1,1].set_xlabel("Measured")
                axes[1,1].set_ylabel("Predicted")
                axes[1,1].set_title("Optimized model - CV")
                axes[1,1].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                                   
                
                #residuals plot
                sns.residplot(self.y_pred_cal_op, self.y - self.y_pred_cal_op, lowess=False, color="k",ax=axes[0, 2])
                axes[0,2].set_xlabel("Predicted")
                axes[0,2].set_ylabel("Residuals")
                axes[0,2].set_title("Optimized model - Cal. Residuals")
                axes[0,2].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                   

                
                sns.residplot(self.y_pred_cv_op, self.y - self.y_pred_cv_op, lowess=False, color="k",ax=axes[1, 2])
                axes[1,2].set_xlabel("Predicted")
                axes[1,2].set_ylabel("Residuals")
                axes[1,2].set_title("Optimized model - CV Residuals")
                axes[1,2].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                      
                
                

                results = {"r2_cal_gen":[self.r2_cal_gen],'rmse_cal_gen':[self.rmse_cal_gen],'rmspe_cal_gen':[self.rmspe_cal_gen],'r2_adj_cal_gen':[self.r2_adj_cal_gen],
                           'r2_cv_gen':[self.r2_cv_gen],'rmse_cv_gen':[self.rmse_cv_gen],'rmspe_cv_gen':[self.rmspe_cv_gen],'r2_adj_cv_gen':[self.r2_adj_cv_gen],
                           "r2_cal_op":[self.r2_cal_op],'rmse_cal_op':[self.rmse_cal_op],'rmspe_cal_op':[self.rmspe_cal_op],'r2_adj_cal_op':[self.r2_adj_cal_op],
                           'r2_cv_op':[self.r2_cv_op],'rmse_cv_op':[self.rmse_cv_op],'rmspe_cv_op':[self.rmspe_cv_op],'r2_adj_cv_op':[self.r2_adj_cv_op]}    
                    
            # if task is classification
            else:
                if len(self.class_sep) == 2:
                    fig,axes = plt.subplots(2,3,figsize=(15,10))
                else:
                    fig,axes = plt.subplots(2,2,figsize=(10 + len(self.class_sep),10 + len(self.class_sep)))
                # general model
                
                self.model.fit(self.x,self.y)
                y_train_pred  = self.model.predict(self.x)
                self.acc_train_gen = self.model.score(self.x,self.y)
                cm = confusion_matrix(self.y,y_train_pred.squeeze())
                cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                sns.heatmap(conf_matrix,cmap="coolwarm", annot=True, ax=axes[0,0],vmin=0,vmax=1)
                axes[0,0].set_title('General model - Cal. confusion matrix')
                axes[0,0].set_ylabel('True label')
                axes[0,0].set_xlabel('Predicted label')            
                                
                
                y_test_pred  = cross_val_predict(self.model, self.x, self.y, 
                             n_jobs = -1,
                             cv=self.cv).squeeze()                  

                if self.cv.__str__() == "LeaveOneOut()":          
                    score_acc = cross_validate_LOO(self.model,self.x,self.y,self.cv,
                                                  scoring=accuracy_score)                                                    

                    self.acc_test_gen = score_acc           
                else:
                    scores = cross_validate(self.model, self.x, self.y, 
                                            n_jobs = -1,
                                            cv=self.cv,
                                            scoring = "accuracy")                    
                    
                    self.acc_test_gen = scores['test_score'].mean()            

                cm = confusion_matrix(self.y,y_test_pred)
                cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                sns.heatmap(conf_matrix,cmap="coolwarm", annot=True,ax = axes[1,0],vmin=0,vmax=1)
                axes[1,0].set_title('General model - CV confusion matrix')  
                axes[1,0].set_ylabel('True label')
                axes[1,0].set_xlabel('Predicted label')           
    
                
                # optimized model
                self.model_best.fit(self.x[:,self.best_features],self.y)
                y_train_pred  = self.model_best.predict(self.x[:,self.best_features])
                self.acc_train_op = self.model_best.score(self.x[:,self.best_features],self.y)
                cm = confusion_matrix(self.y, y_train_pred.squeeze())
                cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                sns.heatmap(conf_matrix,cmap="coolwarm", annot=True, ax=axes[0,1],vmin=0,vmax=1)
                axes[0,1].set_title('Optimized model - Cal. confusion matrix')     
                axes[0,1].set_ylabel('True label')
                axes[0,1].set_xlabel('Predicted label')        
                        
                    
                y_test_pred  = cross_val_predict(self.model_best, self.x[:,self.best_features], self.y, 
                                                 n_jobs = -1,
                                                 cv=self.cv).squeeze()  
                
                if self.cv.__str__() == "LeaveOneOut()":   
                    score_acc = cross_validate_LOO(self.model_best,self.x[:,self.best_features],self.y,
                                                   self.cv,
                                                   scoring=accuracy_score)                        
                    
                    self.acc_test_op = score_acc                                 
                else:
                    scores = cross_validate(self.model_best, self.x[:,self.best_features], self.y, 
                                            n_jobs = -1,
                                            cv=self.cv,
                                            scoring = "accuracy")                              

                    self.acc_test_op = scores['test_score'].mean()
                    
                cm = confusion_matrix(self.y,y_test_pred)
                cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                sns.heatmap(conf_matrix,cmap="coolwarm", annot=True,ax = axes[1,1],vmin=0,vmax=1)
                axes[1,1].set_title('Optimized model - CV confusion matrix')      
                axes[1,1].set_ylabel('True label')
                axes[1,1].set_xlabel('Predicted label')    
                
                if len(self.class_sep) == 2:
                    #optimized model Roc curves
                    self.model_best.fit(self.x[:,self.best_features],self.y)
                    y_train_pred_prob = self.model_best.predict_proba(self.x[:,self.best_features])
                    fpr, tpr, thr = roc_curve(self.y, y_train_pred_prob[:, 1].squeeze())   
                    self.auc_train_op = auc(fpr, tpr)
                    axes[0,2].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
                    axes[0,2].plot(fpr, tpr, color='blue',
                             label=r'$ROC_{train}$ (AUC = %0.2f)' % (auc(fpr, tpr)),lw=2, alpha=1)
                    axes[0,2].set_xlabel('False Positive Rate')
                    axes[0,2].set_ylabel('True Positive Rate')
                    axes[0,2].set_title('Optimized model - Cal. ROC Curve')
                    axes[0,2].legend(loc="lower right")
                    
                    y_test_pred_prob  = cross_val_predict(self.model_best, self.x[:,self.best_features], self.y, 
                                                     n_jobs = -1,
                                                     cv=self.cv,
                                                     method='predict_proba')    
                
                    
                    fpr, tpr, thr = roc_curve(self.y, y_test_pred_prob[:, 1].squeeze())      
                    self.auc_test_op = auc(fpr, tpr)
                    axes[1,2].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
                    axes[1,2].plot(fpr, tpr, color='blue',
                             label=r'$ROC_{test}$ (AUC = %0.2f)' % (auc(fpr, tpr)),lw=2, alpha=1)
                    axes[1,2].set_xlabel('False Positive Rate')
                    axes[1,2].set_ylabel('True Positive Rate')
                    axes[1,2].set_title('Optimized model- CV ROC Curve')
                    axes[1,2].legend(loc="lower right")
                
                    results = {"acc_gen":[self.acc_train_gen],'acc_cv_gen':[self.acc_test_gen],
                       'acc_op':[self.acc_train_op],'acc_cv_op':[self.acc_test_op],
                       "auc_op":[self.auc_train_op],'auc_cv_op':[self.auc_test_op]}
                
                results = {"acc_gen":[self.acc_train_gen],'acc_cv_gen':[self.acc_test_gen],
                   'acc_op':[self.acc_train_op],'acc_cv_op':[self.acc_test_op]}                                
            
        # if cv is not defined
        else:
            if is_regressor(self.model):
                if hasattr(self, 'x_train') and hasattr(self, 'x_test') and hasattr(self, 'y_train') and hasattr(self, 'y_test'):         

                    # calculating accuracies for generalized model
                    self.model.fit(self.x_train,self.y_train)
                    self.y_pred_cal_gen = np.squeeze(self.model.predict(self.x_train)).squeeze()
                    self.r2_cal_gen = r2_score(self.y_train,self.y_pred_cal_gen)
                    #self.r2_adj_cal_gen = adj_r2_score(self.y_train,self.y_pred_cal_gen,x = self.x_train, r2= self.r2_cal_gen)
                    self.r2_adj_cal_gen = adj_r2_score(self.y_train,self.y_pred_cal_gen,x = self.x, r2= self.r2_cal_gen)
                    self.rmse_cal_gen = mean_squared_error(self.y_train, self.y_pred_cal_gen, squared = False)
                    self.rmspe_cal_gen = rmspe_score(self.y_train, self.y_pred_cal_gen)
                    
                    self.y_pred_val_gen = np.squeeze(self.model.predict(self.x_test)).squeeze()
                    self.r2_val_gen = r2_score(self.y_test,self.y_pred_val_gen)
                    #self.r2_adj_val_gen = adj_r2_score(self.y_test, x=self.x_test, r2=self.r2_val_gen)
                    self.r2_adj_val_gen = adj_r2_score(self.y_test, x=self.x, r2=self.r2_val_gen)
                    self.rmse_val_gen = mean_squared_error(self.y_test, self.y_pred_val_gen, squared = False)
                    self.rmspe_val_gen = rmspe_score(self.y_test, self.y_pred_val_gen)
                    
                    # calculating accuracies for optimized model
                    self.model_best.fit(self.x_train[:,self.best_features],self.y_train)
                    self.y_pred_cal_op = np.squeeze(self.model_best.predict(self.x_train[:,self.best_features])).squeeze()
                    self.r2_cal_op = r2_score(self.y_train,self.y_pred_cal_op)
                    #self.r2_adj_cal_op = adj_r2_score(self.y_train, x=self.x_train[:,self.best_features],r2=self.r2_cal_op)
                    self.r2_adj_cal_op = adj_r2_score(self.y_train, x=self.x[:,self.best_features],r2=self.r2_cal_op)
                    self.rmse_cal_op = mean_squared_error(self.y_train, self.y_pred_cal_op, squared = False)  
                    self.rmspe_cal_op = rmspe_score(self.y_train, self.y_pred_cal_op)
                    
                    self.y_pred_val_op = np.squeeze(self.model_best.predict(self.x_test[:,self.best_features])).squeeze()
                    self.r2_val_op = r2_score(self.y_test,self.y_pred_val_op)
                    #self.r2_adj_val_op = adj_r2_score(self.y_test, x= self.x_test[:,self.best_features],r2=self.r2_val_op)
                    self.r2_adj_val_op = adj_r2_score(self.y_test, x= self.x[:,self.best_features],r2=self.r2_val_op)
                    self.rmse_val_op = mean_squared_error(self.y_test, self.y_pred_val_op, squared = False)
                    self.rmspe_val_op = rmspe_score(self.y_test, self.y_pred_val_op)
                    
                    # generalized model
                    fig,axes = plt.subplots(2,3,figsize=(15,10))
                    sns.regplot(x=self.y_train, y=self.y_pred_cal_gen, color="k",ax=axes[0, 0])
                    axes[0,0].plot([np.min([self.y_train, self.y_pred_cal_gen]) - 0.1*np.std(self.y), np.max([self.y_train, self.y_pred_cal_gen]) + 0.1*np.std(self.y)],
                        [np.min([self.y_train, self.y_pred_cal_gen]) - 0.1*np.std(self.y), np.max([self.y_train, self.y_pred_cal_gen]) + 0.1*np.std(self.y)],'r--')
                    axes[0,0].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_cal_gen) + "\n"+ 
                                       r"RMSPE={:.3f}".format(self.rmspe_cal_gen) + "\n"+ 
                                      r"$R^2$={:.3f}".format(self.r2_cal_gen)+ "\n"+
                                      r"$Adj. R^2$={:.3f}".format(self.r2_adj_cal_gen),
                                      transform=axes[0, 0].transAxes,
                                      color='k',
                                      bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                    axes[0,0].set_xlabel("Measured")
                    axes[0,0].set_ylabel("Predicted")
                    axes[0,0].set_title("General model - Cal.")
                    axes[0,0].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                                       
                    
                    sns.regplot(x=self.y_test, y=self.y_pred_val_gen, color="k",ax=axes[1, 0])
                    axes[1,0].plot([np.min([self.y_test, self.y_pred_val_gen]) - 0.1*np.std(self.y), np.max([self.y_test, self.y_pred_val_gen]) + 0.1*np.std(self.y)],
                        [np.min([self.y_test, self.y_pred_val_gen]) - 0.1*np.std(self.y), np.max([self.y_test, self.y_pred_val_gen]) + 0.1*np.std(self.y)],'r--')            
                    axes[1,0].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_val_gen) + "\n"+ 
                                       r"RMSPE={:.3f}".format(self.rmspe_val_gen) + "\n"+ 
                                      r"$R^2$={:.3f}".format(self.r2_val_gen)+ "\n"+
                                      r"$Adj. R^2$={:.3f}".format(self.r2_adj_val_gen),
                                      transform=axes[1, 0].transAxes,
                                      color='k',
                                      bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                    axes[1,0].set_xlabel("Measured")
                    axes[1,0].set_ylabel("Predicted")
                    axes[1,0].set_title("General model - Val.")
                    axes[1,0].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                               

                    # optimized model
                    sns.regplot(x=self.y_train, y=self.y_pred_cal_op, color="k",ax=axes[0, 1])
                    axes[0,1].plot([np.min([self.y_train, self.y_pred_cal_op]) - 0.1*np.std(self.y), np.max([self.y_train, self.y_pred_cal_op]) + 0.1*np.std(self.y)],
                        [np.min([self.y_train, self.y_pred_cal_op]) - 0.1*np.std(self.y), np.max([self.y_train, self.y_pred_cal_op]) + 0.1*np.std(self.y)],'r--')               
                    axes[0,1].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_cal_op) + "\n"+ 
                                       r"RMSPE={:.3f}".format(self.rmspe_cal_op) + "\n"+ 
                                      r"$R^2$={:.3f}".format(self.r2_cal_op)+ "\n"+
                                      r"$Adj. R^2$={:.3f}".format(self.r2_adj_cal_op),
                                      transform=axes[0, 1].transAxes,
                                      color='k',
                                      bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                    axes[0,1].set_xlabel("Measured")
                    axes[0,1].set_ylabel("Predicted")
                    axes[0,1].set_title("Optimized model - Cal.")
                    axes[0,1].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                                                         
                    
                    sns.regplot(x=self.y_test, y=self.y_pred_val_op, color="k",ax=axes[1, 1])
                    axes[1,1].plot([np.min([self.y_test, self.y_pred_val_op]) - 0.1*np.std(self.y), np.max([self.y_test, self.y_pred_val_op]) + 0.1*np.std(self.y)],
                        [np.min([self.y_test, self.y_pred_val_op]) - 0.1*np.std(self.y), np.max([self.y_test, self.y_pred_val_op]) + 0.1*np.std(self.y)],'r--')              
                    axes[1,1].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_val_op) + "\n"+ 
                                      r"RMSPE={:.3f}".format(self.rmspe_val_op) + "\n"+ 
                                      r"$R^2$={:.3f}".format(self.r2_val_op)+ "\n"+
                                      r"$Adj. R^2$={:.3f}".format(self.r2_adj_val_op),
                                      transform=axes[1, 1].transAxes,
                                      color='k',
                                      bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                    axes[1,1].set_xlabel("Measured")
                    axes[1,1].set_ylabel("Predicted")
                    axes[1,1].set_title("Optimized model - Val.")
                    axes[1,1].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                               
                    
                    #residuals plot
                    sns.residplot(self.y_pred_cal_op, self.y_train - self.y_pred_cal_op, lowess=False, color="k",ax=axes[0, 2])
                    axes[0,2].set_xlabel("Predicted")
                    axes[0,2].set_ylabel("Residuals")
                    axes[0,2].set_title("Optimized model - Cal. Residuals")
                    axes[0,2].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                     
                    
                    
                    sns.residplot(self.y_pred_val_op, self.y_test - self.y_pred_val_op, lowess=False, color="k",ax=axes[1, 2])
                    axes[1,2].set_xlabel("Predicted")
                    axes[1,2].set_ylabel("Residuals")
                    axes[1,2].set_title("Optimized model - Val. Residuals")        
                    axes[1,2].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                                       
                    
                    results = {"r2_cal_gen":[self.r2_cal_gen],'rmse_cal_gen':[self.rmse_cal_gen], 'rmspe_cal_gen':[self.rmspe_cal_gen], 'r2_adj_cal_gen':[self.r2_adj_cal_gen],
                               'r2_val_gen':[self.r2_val_gen],'rmse_val_gen':[self.rmse_val_gen],'rmspe_val_gen':[self.rmspe_val_gen], 'r2_adj_val_gen':[self.r2_adj_val_gen],
                               "r2_cal_op":[self.r2_cal_op],'rmse_cal_op':[self.rmse_cal_op],'rmspe_cal_op':[self.rmspe_cal_op], 'r2_adj_cal_op':[self.r2_adj_cal_op],
                               'r2_val_op':[self.r2_val_op],'rmse_val_op':[self.rmse_val_op],'rmspe_val_op':[self.rmspe_val_op], 'r2_adj_val_op':[self.r2_adj_val_op]}      
                    
                # test partitions not defined 
                else:
                    # calculating accuracies for generalized model
                    self.model.fit(self.x,self.y)
                    self.y_pred_cal_gen = np.squeeze(self.model.predict(self.x)).squeeze()
                    self.r2_cal_gen = r2_score(self.y,self.y_pred_cal_gen)
                    self.r2_adj_cal_gen = adj_r2_score(self.y,x=self.x, r2=self.r2_cal_gen)
                    self.rmse_cal_gen = mean_squared_error(self.y, self.y_pred_cal_gen, squared = False)    
                    self.rmspe_cal_gen = rmspe_score(self.y, self.y_pred_cal_gen)    
                    
                    # calculating accuracies for optimized model
                    self.model_best.fit(self.x[:,self.best_features],self.y)
                    self.y_pred_cal_op = np.squeeze(self.model_best.predict(self.x[:,self.best_features])).squeeze()
                    self.r2_cal_op = r2_score(self.y,self.y_pred_cal_op)
                    self.r2_adj_cal_op = adj_r2_score(self.y,x=self.x[:,self.best_features], r2=self.r2_cal_op)
                    self.rmse_cal_op = mean_squared_error(self.y, self.y_pred_cal_op, squared = False)  
                    self.rmspe_cal_op = rmspe_score(self.y, self.y_pred_cal_op)    
                    
                    # generalized model
                    fig,axes = plt.subplots(1,3,figsize=(15,5))
                    sns.regplot(x=self.y, y=self.y_pred_cal_gen, color="k",ax=axes[0])
                    axes[0].plot([np.min([self.y, self.y_pred_cal_gen]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cal_gen]) + 0.1*np.std(self.y)],
                        [np.min([self.y, self.y_pred_cal_gen]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cal_gen]) + 0.1*np.std(self.y)],'r--')
                    axes[0].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_cal_gen) + "\n"+ 
                                     r"RMSPE={:.3f}".format(self.rmspe_cal_gen) + "\n"+ 
                                      r"$R^2$={:.3f}".format(self.r2_cal_gen)+ "\n"+
                                      r"$Adj. R^2$={:.3f}".format(self.r2_adj_cal_gen),
                                      transform=axes[0].transAxes,
                                      color='k',
                                      bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                    axes[0].set_xlabel("Measured")
                    axes[0].set_ylabel("Predicted")
                    axes[0].set_title("General model - Cal.")
                    axes[0].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                                                          
                    
                    
                    # optimized model
                    sns.regplot(x=self.y, y=self.y_pred_cal_op, color="k",ax=axes[1])
                    axes[1].text(0.05,0.80,r"RMSE={:.3f}".format(self.rmse_cal_op) + "\n"+ 
                                     r"RMSPE={:.3f}".format(self.rmspe_cal_op) + "\n"+ 
                                      r"$R^2$={:.3f}".format(self.r2_cal_op)+ "\n"+
                                      r"$Adj. R^2$={:.3f}".format(self.r2_adj_cal_op),
                                      transform=axes[1].transAxes,
                                      color='k',
                                      bbox=dict(facecolor=(1,1,1,0.5),edgecolor=(0,0,0,1)))
                    axes[1].plot([np.min([self.y, self.y_pred_cal_op]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cal_op]) + 0.1*np.std(self.y)],
                        [np.min([self.y, self.y_pred_cal_op]) - 0.1*np.std(self.y), np.max([self.y, self.y_pred_cal_op]) + 0.1*np.std(self.y)],'r--')
                    axes[1].set_xlabel("Measured")
                    axes[1].set_ylabel("Predicted")
                    axes[1].set_title("Optimized model - Cal.")
                    axes[1].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                                       
                    plt.show(block=False)
                    
                    sns.residplot(self.y_pred_cal_op, self.y - self.y_pred_cal_op, lowess=False, color="r",ax=axes[2])
                    axes[2].set_xlabel("Predicted")
                    axes[2].set_ylabel("Residuals")
                    axes[2].set_title("Optimized model - Cal. Residuals") 
                    axes[2].ticklabel_format(axis="both", style = "sci", scilimits = (0,2))                                                       

                    results = {"r2_cal_gen":[self.r2_cal_gen],'rmse_cal_gen':[self.rmse_cal_gen],'rmspe_cal_gen':[self.rmspe_cal_gen],"r2_adj_cal_gen":[self.r2_adj_cal_gen],
                                   "r2_cal_op":[self.r2_cal_op],'rmse_cal_op':[self.rmse_cal_op],'rmspe_cal_op':[self.rmspe_cal_op], "r2_adj_cal_op":[self.r2_adj_cal_op]}                          
            
            # if the task is classification
            else:
                if hasattr(self, 'x_train') and hasattr(self, 'x_test') and hasattr(self, 'y_train') and hasattr(self, 'y_test'):         
                    if len(self.class_sep) == 2:
                        fig,axes = plt.subplots(2,3,figsize=(15,10))
                    else:
                        fig,axes = plt.subplots(2,2,figsize=(10 + len(self.class_sep),10 + len(self.class_sep)))
                    # general model
                    self.model.fit(self.x_train,self.y_train)
                    y_train_pred  = self.model.predict(self.x_train)
                    self.acc_train_gen = self.model.score(self.x_train,self.y_train)
                    cm = confusion_matrix(self.y_train,y_train_pred.squeeze())
                    cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                    conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                    sns.heatmap(conf_matrix,cmap="coolwarm", annot=True, ax=axes[0,0],vmin=0,vmax=1)
                    axes[0,0].set_title('General model - Cal. confusion matrix')
                    axes[0,0].set_ylabel('True label')
                    axes[0,0].set_xlabel('Predicted label')

                    self.model.fit(self.x_train,self.y_train)
                    y_test_pred  = self.model.predict(self.x_test)
                    self.acc_test_gen = self.model.score(self.x_test,self.y_test)
                    cm = confusion_matrix(self.y_test,y_test_pred.squeeze())
                    cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                    conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                    sns.heatmap(conf_matrix,cmap="coolwarm", annot=True,ax = axes[1,0],vmin=0,vmax=1)
                    axes[1,0].set_title('General model - Val. confusion matrix')  
                    axes[1,0].set_ylabel('True label')
                    axes[1,0].set_xlabel('Predicted label')
        
                    # optimized model
                    self.model_best.fit(self.x_train[:,self.best_features],self.y_train)
                    y_train_pred  = self.model_best.predict(self.x_train[:,self.best_features])
                    self.acc_train_op = self.model_best.score(self.x_train[:,self.best_features],self.y_train)
                    cm = confusion_matrix(self.y_train,y_train_pred.squeeze())
                    cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                    conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                    sns.heatmap(conf_matrix,cmap="coolwarm", annot=True, ax=axes[0,1],vmin=0,vmax=1)
                    axes[0,1].set_title('Optimized model - Cal. confusion matrix')     
                    axes[0,1].set_ylabel('True label')
                    axes[0,1].set_xlabel('Predicted label')                
                    
                    self.model_best.fit(self.x_train[:,self.best_features],self.y_train)
                    y_test_pred  = self.model_best.predict(self.x_test[:,self.best_features])
                    self.acc_test_op = self.model_best.score(self.x_test[:,self.best_features],self.y_test)
                    cm = confusion_matrix(self.y_test,y_test_pred.squeeze())
                    cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                    conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                    sns.heatmap(conf_matrix,cmap="coolwarm", annot=True,ax = axes[1,1],vmin=0,vmax=1)
                    axes[1,1].set_title('Optimized model - Val. confusion matrix')      
                    axes[1,1].set_ylabel('True label')
                    axes[1,1].set_xlabel('Predicted label')
        
                    if len(self.class_sep) == 2:
                        #optimized model Roc curves
                        self.model_best.fit(self.x_train[:,self.best_features],self.y_train)
                        y_train_pred_prob = self.model_best.predict_proba(self.x_train[:,self.best_features])
                        fpr, tpr, thr = roc_curve(self.y_train, y_train_pred_prob[:, 1].squeeze())   
                        self.auc_train_op = auc(fpr, tpr)
                        axes[0,2].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
                        axes[0,2].plot(fpr, tpr, color='blue',
                                 label=r'$ROC_{train}$ (AUC = %0.2f)' % (auc(fpr, tpr)),lw=2, alpha=1)
                        axes[0,2].set_xlabel('False Positive Rate')
                        axes[0,2].set_ylabel('True Positive Rate')
                        axes[0,2].set_title('ROC Curve')
                        axes[0,2].legend(loc="lower right")
                        
                        self.model_best.fit(self.x_train[:,self.best_features],self.y_train)
                        y_test_pred_prob = self.model_best.predict_proba(self.x_test[:,self.best_features])
                        fpr, tpr, thr = roc_curve(self.y_test, y_test_pred_prob[:, 1].squeeze())      
                        self.auc_test_op = auc(fpr, tpr)
                        axes[1,2].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
                        axes[1,2].plot(fpr, tpr, color='blue',
                                 label=r'$ROC_{test}$ (AUC = %0.2f)' % (auc(fpr, tpr)),lw=2, alpha=1)
                        axes[1,2].set_xlabel('False Positive Rate')
                        axes[1,2].set_ylabel('True Positive Rate')
                        axes[1,2].set_title('ROC Curve')
                        axes[1,2].legend(loc="lower right")
                    
                        results = {"acc_train_gen":[self.acc_train_gen],'acc_test_gen':[self.acc_test_gen],
                           'acc_train_op':[self.acc_train_op],'acc_test_op':[self.acc_test_op],
                           "auc_train_op":[self.auc_train_op],'auc_test_op':[self.auc_test_op]}
                    
                    results = {"acc_train_gen":[self.acc_train_gen],'acc_test_gen':[self.acc_test_gen],
                       'acc_train_op':[self.acc_train_op],'acc_test_op':[self.acc_test_op]}
                
                # if train and test not defined
                else:
                    warnings.warn("Neither cv_model nor test partition (x_test/y_test) are defined, predicting on x/y.")
                    if len(self.class_sep) == 2:
                        fig,axes = plt.subplots(1,3,figsize=(15,5))
                    else:
                        fig,axes = plt.subplots(1,2,figsize=(10 + len(self.class_sep),5 + len(self.class_sep)))                        
                    # general model
                    self.model.fit(self.x,self.y)
                    y_train_pred  = self.model.predict(self.x)
                    self.acc_train_gen = self.model.score(self.x,self.y)
                    cm = confusion_matrix(self.y,y_train_pred.squeeze())
                    cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                    conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                    sns.heatmap(conf_matrix,cmap="coolwarm", annot=True, ax=axes[0],vmin=0,vmax=1)
                    axes[0].set_title('General model - Cal. confusion matrix')
                    axes[0].set_ylabel('True label')
                    axes[0].set_xlabel('Predicted label')
        
                    # optimized model
                    self.model_best.fit(self.x[:,self.best_features],self.y)
                    y_train_pred  = self.model_best.predict(self.x[:,self.best_features])
                    self.acc_train_op = self.model_best.score(self.x[:,self.best_features],self.y)
                    cm = confusion_matrix(self.y,y_train_pred.squeeze())
                    cm = np.stack(list(map(lambda x: x/x.sum(),cm)))
                    conf_matrix=pd.DataFrame(cm,columns = self.class_sep,index= self.class_sep)
                    sns.heatmap(conf_matrix,cmap="coolwarm", annot=True, ax=axes[1],vmin=0,vmax=1)
                    axes[1].set_title('Optimized model - Cal. confusion matrix')     
                    axes[1].set_ylabel('True label')
                    axes[1].set_xlabel('Predicted label')                
        
        
                    if len(self.class_sep) == 2:
                        #optimized model Roc curves
                        self.model_best.fit(self.x[:,self.best_features],self.y)
                        y_train_pred_prob = self.model_best.predict_proba(self.x[:,self.best_features])
                        fpr, tpr, thr = roc_curve(self.y, y_train_pred_prob[:, 1].squeeze())   
                        self.auc_train_op = auc(fpr, tpr)
                        axes[2].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
                        axes[2].plot(fpr, tpr, color='blue',
                                 label=r'$ROC_{train}$ (AUC = %0.2f)' % (auc(fpr, tpr)),lw=2, alpha=1)
                        axes[2].set_xlabel('False Positive Rate')
                        axes[2].set_ylabel('True Positive Rate')
                        axes[2].set_title('ROC Curve')
                        axes[2].legend(loc="lower right")
    
                    
                        results = {"acc_train_gen":[self.acc_train_gen],
                           'acc_train_op':[self.acc_train_op],
                           "auc_train_op":[self.auc_train_op]}
                    
                    results = {"acc_train_gen":[self.acc_train_gen],
                       'acc_train_op':[self.acc_train_op]}
                    
        plt.show(block=False) 
        return results, fig

    def tune_model_params(self,index, param_dict,random_state = 1):
        self.best_features = literal_eval(self.res_df.iloc[index]["Feauteres_decor_ind"])
        a = time.time()
        Searcher = RandomizedSearchCV(self.model, param_dict, random_state=random_state,n_jobs=-1, scoring = self.scoring)
        if hasattr(self, 'x_train') and hasattr(self, 'x_test') and hasattr(self, 'y_train') and hasattr(self, 'y_test'):
            search = Searcher.fit(self.x_train[:,self.best_features],self.y_train)
        else:
            search = Searcher.fit(self.x[:,self.best_features],self.y)
        if self.verbose:
            print("Parameter tuning completed in {:.2f} seconds".format(time.time() - a))            
            print("best params are: \n",search.best_params_)
        self.model_best = search.best_estimator_
        
        return search.best_estimator_
        
    @property
    def _name_(self):
        return "NSGA2"        
        