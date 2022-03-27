# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:27:57 2020

@author: Amirh
"""
from ..metrics._metrics import adj_r2_score, kappa_score, rmse_score, rmspe_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.base import is_regressor, is_classifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import *
from sklearn.feature_selection import VarianceThreshold, f_regression, f_classif, \
    mutual_info_regression, mutual_info_classif, chi2
from scipy import stats
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import type_of_target, unique_labels
from ..metrics._metrics import cross_validate_LOO
from scipy.optimize import minimize
from sklearn.inspection import permutation_importance
from copy import copy
from functools import partial, update_wrapper

class BaseFeatureSelector(object):
    def __init__(self,n_f=None,scoring = None,verbose = False,**kwargs):
        self.n_f = n_f
        self.scoring = scoring
        # check scoring
        #self.scorer = self._check_scoring()   
        self.scorer = self._scorer_maker(**kwargs)        
        self.verbose = verbose

        self.x_test, self.y_test = None, None
    
    def load(self,x,y,decor=None,scale=False):
        """
        This function loads the data, decorrellates and scales the given data.
        """
        
        # load data    
        x = check_array(x)           
        x,y = check_X_y(x,y)
        self.x = x
                
        # checking to see if the model is regression or classification
        if is_regressor(self.model) and (type_of_target(y) == "continuous" or 
                                         type_of_target(y) == "multiclass"): #for floats that are whole numbers
            self.y = y
        elif (is_classifier(self.model)) and (type_of_target(y) == "multiclass" or 
                                              type_of_target(y) == "binary"):
            self.y = y
            self.class_sep = unique_labels(self.y).astype(np.int16)   

        else:
            raise ValueError("Mismatch between objective model and the input data.")                     
        if self.verbose:            
            print("Data successfully loaded.")  
        
        # scale data
        if scale:
            scaler = StandardScaler()
            self.x = scaler.fit_transform(self.x)
            if self.verbose:
                print("Data standardized")    
                
        # decorrellate data
        if decor is not None:
            x_df = pd.DataFrame(self.x)
            if type(decor) is tuple:
                indices = self._feature_importance(decor[0])                
                x_df = x_df[indices] # sorting features
                decor = decor[1]
                
            if decor == int(decor):
                x_df = self._decorrellate_n(x_df, decor)
                self.x_cols = x_df.columns    
                self.x = x_df.values
            
            if decor == float(decor):
                x_df = self._decorrellate_thr(x_df, decor)
                self.x_cols = x_df.columns    
                self.x = x_df.values
                
            
            else:
	            raise ValueError('Decorrellation value should be either int for number of features or float for pearson decorrellation.')


            if self.verbose:
                print("Data decorrellated to {} features.".format(self.x.shape[1]))
        else:
        	self.x_cols = pd.DataFrame(self.x).columns    

            
    def train_test_split(self, test_size, random_state=1,**kwargs):
        """
        The function splits the data into train and test size.        
        """
        
        if self.cv:
            raise ValueError("Either train/test partition or cv should be defined.")     
        if test_size > 0 and test_size < 1:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=random_state,**kwargs)
        else:
            raise ValueError('test_size should be bigger than 0 and smaller than 1.') 
        if self.verbose:            
            print("train/test split created with test size of {}.".format(test_size))            

    
    def tune_model_params(self,param_dict,random_state = 1):
        """
        Function tuning model parameters using RandomizedCVSearch.

        Parameters
        ----------
        param_dict : dictionary
            Dictionary containing parameters of the given model.
        random_state : int, optional
            Seed for random state. The default is 1.

        Returns
        -------
        Estimator
            Function returns the best estimator chosen by RandomizedCVSearch.

        """        
        a = time.time()
        Searcher = RandomizedSearchCV(self.model, param_dict, random_state=random_state,n_jobs=-1)
        if hasattr(self, 'x_train') and hasattr(self, 'x_test') and hasattr(self, 'y_train') and hasattr(self, 'y_test'):
            search = Searcher.fit(self.x_train[:,self.best_features],self.y_train)
        else:
            search = Searcher.fit(self.x[:,self.best_features],self.y)
        if self.verbose:
            print("Parameter tuning completed in {:.2f} seconds".format(time.time() - a))            
            print("best params are: \n",search.best_params_)
        self.model_best = search.best_estimator_
        
        return search.best_estimator_

    def display_results(self):
        """
        Function displaying results for the best identified set of features.

        Returns
        -------
        Accuracy metrics for the best identified set of features.

        """
        
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
      
    def _decorrellate_thr(self,x_df, threshold):        
        dataset = x_df.copy()
        col_corr = set() # Set of all the names of deleted columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (np.abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column
                    col_corr.add(colname)
                    if colname in dataset.columns:
                        del dataset[colname] # deleting the column from the dataset
        new_dataset = dataset.copy()
        return new_dataset        

    def _decorrellate_n(self,x_df, n):        
        # optimizing the decorrellate_thresh to find the threshold that gives out number of features matching with that
        x0 = [0.9]
        func = lambda x: np.abs(len(self._decorrellate_thr(x_df,x).columns) - n) #minimizng this function 
        thresh = minimize(func, x0, method = "Nelder-Mead").x[0]        
        new_dataset = self._decorrellate_thr(x_df,thresh)
        self.decor_thresh = thresh        
        return new_dataset              
    
    def _scorer_maker(self,**kwargs):
        scorer_func = partial(self.scoring,**kwargs)
        update_wrapper(scorer_func, self.scoring)
        return scorer_func
    
    def _feature_importance(self,approach):
        
        valid_approaches = ["variance",
                            "f_test",
                            "mutual_info",
                            "chi2",
                            "pearson"]
        if approach not in valid_approaches:
            raise ValueError(f"Valid approaches are {valid_approaches}.")
    
        if approach == "variance":
            selector = VarianceThreshold()
            selector.fit(self.x)
            var = selector.variances_
            indices = np.argsort(var)[::-1]
            
        elif approach == "f_test":
            if is_regressor(self.model):
                f,_ = f_regression(self.x,self.y)
                indices = np.argsort(f)[::-1]
            else:
                f,_ = f_classif(self.x,self.y)
                indices = np.argsort(f)[::-1]
                            
        elif approach == "mutual_info":
            if is_regressor(self.model):
                mi = mutual_info_regression(self.x,self.y)
                indices = np.argsort(mi)[::-1]
            else:
                mi = mutual_info_classif(self.x,self.y)
                indices = np.argsort(mi)[::-1]            
            
        elif approach == "chi2":            
            if is_regressor(self.model):
                raise ValueError("chi2 is only available for classification.")
            else:
                chi_sqaured,_ = chi2(self.x,self.y)
                indices = np.argsort(chi_sqaured)[::-1]            
                
        elif approach == "pearson":
            if is_regressor(self.model):            
                r = np.abs(np.array([stats.pearsonr(self.x[:,i],self.y)[0] for i in range(self.x.shape[1])]))
                indices = np.argsort(r)[::-1]        
            else:
                raise ValueError("pearson is only available for regression.")
                            
        return indices
    
    def _feature_ranking(self, tops,columns, n_repeats, random_state):
        # here we are multiplying the initial permutation feature importance by the weight acquired
        # from the top 25% percentile agents and their occurence ratio
        top_agents, top_perc = tops
        
        # getting w        
        top_agents_arr = np.concatenate(top_agents)
        indices, counts = np.unique(top_agents_arr,return_counts=True) #indices are all features in the pool, and counts is the number of appearance totally
        w = counts/len(top_agents)
        
        # getting alpha, which is based on percentage score (rmspe or f1)
        # first creating a mask to see where features belong
        indices_mask = np.zeros((len(indices),len(top_agents)),dtype=bool)        
        for c1, index in enumerate(indices):    
            for c2, agent in enumerate(top_agents):
                indices_mask[c1,c2] = index in agent 
                
        # getting mean perc based on the mask for each index (feature)        
        alpha = np.zeros(w.shape)
        for c, index_mask in enumerate(indices_mask):
            alpha[c] = np.mean(np.array(top_perc)[index_mask])                    
        
        model = self.model_best.fit(self.x,self.y)
        rankings = permutation_importance(model, self.x, self.y,n_repeats=n_repeats,random_state=random_state).importances_mean        
        new_rankings = copy(rankings)
                
        for c,index in enumerate(indices):    
            if is_regressor(model): #using rmspe
                new_rankings[index] = new_rankings[index]*(1 + (1-alpha[c])*w[c])
            else: #using f1
                new_rankings[index] = new_rankings[index]*(1 + alpha[c]*w[c])
        
        new_rankings = np.concatenate([np.expand_dims(columns,0), 
                                       np.expand_dims(new_rankings,0)],axis=0)
        
        return new_rankings 
         
    def _check_scoring(self):
        raise DeprecationWarning("The function is deprecated and will be removed in future versions. _scorer_maker replaces this function.")
        
        # setting up scoring
        score_valid = {
        # classification accuracies
        "accuracy": accuracy_score,
        "balanced_accuracy":balanced_accuracy_score,
        "average_precision":average_precision_score,
        "f1":f1_score,
        "neg_brier_score":brier_score_loss,
        "neg_log_loss":log_loss,
        "precision":precision_score,
        "recall":recall_score,
        "jaccard":jaccard_score,
        "kappa":kappa_score,
        # regression accuracies 
        "explained_variance":explained_variance_score,
        "max_error":max_error,
        "mean_absolute_error":mean_absolute_error,
        "mean_squared_error":mean_squared_error,
        "root_mean_squared_error":rmse_score,
        "mean_squared_log_error":mean_squared_log_error,
        "median_absolute_error":median_absolute_error,
        "r2":r2_score,
        "adj_r2":adj_r2_score,
        "mean_poisson_deviance":mean_poisson_deviance,
        "mean_gamma_deviance":mean_gamma_deviance,
        "mean_tweedie_deviance":mean_tweedie_deviance}
        
        
        # the function takes in a string and returns a scorer
        if self.scoring in score_valid.keys():
            scorer = score_valid[self.scoring]
        else:
            raise ValueError(f"{self.scoring} is not a valid scorer. Valid scorers are:\n {score_valid.keys()}")
                
        return scorer    
         
        
        
        
        
        
        
        
        
        
        
        
        
        
        

