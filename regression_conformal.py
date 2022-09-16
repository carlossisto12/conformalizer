# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:31:46 2022

@author: Carlos Sisto
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  Ridge, Lasso, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,  AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm
from joblib import Parallel, delayed
import os
warnings.filterwarnings('ignore')

class ConformalRegressor:
    def __init__(self, df, target, n_calib, random_state=42, test_size=0.2, scaler=None):
        self.df = df
        self.target = target
        self.n_calib = n_calib
        self.random_state = random_state
        self.test_size = test_size
        self.scaler = scaler
        self.calib = self.df.sample(n=self.n_calib, random_state = self.random_state)
        self.df = self.df[~self.df.index.isin(self.calib.index)]
        x = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        self.x_calib = self.calib.drop(self.target, axis=1)
        self.y_calib = self.calib[self.target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,
                                                                                y,
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)
        if self.scaler is not None:
            self.x_train = self.scaler.fit_transform(self.x_train)
            self.x_test= self.scaler.transform(self.x_test)
            self.x_calib = self.scaler.transform(self.x_calib)
        
        self.calib_ = pd.concat([pd.DataFrame(self.x_calib,
                                              columns = self.df.drop(columns=self.target).columns),
                                 pd.DataFrame(self.y_calib).reset_index(drop=True)],
                                axis = 1)
        self.test_ = pd.concat([pd.DataFrame(self.x_test,
                                              columns = self.df.drop(columns=self.target).columns),
                                 pd.DataFrame(self.y_test).reset_index(drop=True)],
                                axis = 1)
        
    def _compute_qhat(self, calib_pred, alpha):
        scores = calib_pred[['t_inf-y', 'y-t_sup']].max(axis=1).to_numpy()
        q = np.ceil((self.n_calib+1)*(1-alpha))/self.n_calib
        qhat = np.quantile(scores, q)
        return qhat
    
    def linear_conformal_regression(self, formula, alpha):
        model_inf = smf.quantreg(formula, self.calib_).fit(q=alpha/2)
        model_sup = smf.quantreg(formula, self.calib_).fit(q=1-alpha/2)
        
        model_inf_test = smf.quantreg(formula, self.test_).fit(q=alpha/2)
        model_sup_test = smf.quantreg(formula, self.test_).fit(q=1-alpha/2)
        
        calib_pred = pd.DataFrame(self.calib_[self.target])
        calib_pred['t_inf'] = model_inf.predict(self.calib_.drop(columns=self.target))
        calib_pred['t_sup'] = model_sup.predict(self.calib_.drop(columns=self.target))
        calib_pred['t_inf-y'] = -calib_pred[self.target] + calib_pred['t_inf']
        calib_pred['y-t_sup'] = calib_pred[self.target] - calib_pred['t_sup']
        
        qhat = self._compute_qhat(calib_pred, alpha)
        
        predictions = pd.DataFrame(self.test_[self.target])
        predictions['lower_linear_conformal'] = model_inf_test.predict(self.test_.drop(columns=self.target)) - qhat
        predictions['upper_linear_conformal'] = model_sup_test.predict(self.test_.drop(columns=self.target)) + qhat
        return predictions
    
    def random_forest_conformal_regression(self, alpha, parameters=None):
        if parameters is not None:
            rf = RandomForestRegressor(**parameters)
            rf_test = RandomForestRegressor(**parameters)
        else:
            rf = RandomForestRegressor()
            rf_test = RandomForestRegressor()
        
        rf.fit(self.calib_.drop(columns=self.target), self.calib_[self.target])
        calib_pred = pd.DataFrame()
        for pred in tqdm(rf.estimators_):
            temp = pd.Series(pred.predict(self.calib_.drop(columns=self.target)))
            calib_pred = pd.concat([calib_pred, temp], axis = 1)
        
        calib_pred = pd.DataFrame(self.calib_[self.target])   
        calib_pred['t_inf'] = calib_pred.quantile(alpha/2, axis = 1)
        calib_pred['t_sup'] = calib_pred.quantile(1-alpha/2, axis = 1)
        calib_pred['t_inf-y'] = -calib_pred[self.target] + calib_pred['t_inf']
        calib_pred['y-t_sup'] = calib_pred[self.target] - calib_pred['t_sup']
        
        qhat = self._compute_qhat(calib_pred, alpha)
        
        rf_test.fit(self.test_.drop(columns=self.target), self.test_[self.target])
        pred_rf_test = pd.DataFrame()
        for pred in tqdm(rf_test.estimators_):
            temp = pd.Series(pred.predict(self.test_.drop(columns=self.target)))
            pred_rf_test = pd.concat([pred_rf_test, temp], axis = 1)
            
        predictions = pd.DataFrame(self.test_[self.target])
        predictions['lower_random_forest_conformal'] = pred_rf_test.quantile(alpha/2, axis = 1) - qhat
        predictions['upper_random_forest_conformal'] = pred_rf_test.quantile(1-alpha/2, axis = 1) + qhat
        return predictions
        
    def gradient_boosting_conformal_regression(self, alpha, parameters=None):
        if parameters is not None:
            gb_inf = GradientBoostingRegressor(loss = 'quantile', alpha = alpha/2, **parameters)
            gb_inf_test = GradientBoostingRegressor(loss = 'quantile', alpha = alpha/2, **parameters)
            gb_sup = GradientBoostingRegressor(loss = 'quantile', alpha = 1-alpha/2, **parameters)
            gb_sup_test = GradientBoostingRegressor(loss = 'quantile', alpha = 1-alpha/2, **parameters)
        else:
            gb_inf = GradientBoostingRegressor(loss = 'quantile', alpha = alpha/2)
            gb_inf_test = GradientBoostingRegressor(loss = 'quantile', alpha = alpha/2)
            gb_sup = GradientBoostingRegressor(loss = 'quantile', alpha = 1-alpha/2)
            gb_sup_test = GradientBoostingRegressor(loss = 'quantile', alpha = 1-alpha/2)
            
        gb_inf.fit(self.calib_.drop(columns=self.target), self.calib_[self.target])
        gb_sup.fit(self.calib_.drop(columns=self.target), self.calib_[self.target])
        
        calib_pred = pd.DataFrame(self.calib_[self.target])
        calib_pred['t_inf'] = gb_inf.predict(self.calib_.drop(columns=self.target))
        calib_pred['t_sup'] = gb_sup.predict(self.calib_.drop(columns=self.target))
        calib_pred['t_inf-y'] = -calib_pred[self.target] + calib_pred['t_inf']
        calib_pred['y-t_sup'] = calib_pred[self.target] - calib_pred['t_sup']
        
        qhat = self._compute_qhat(calib_pred, alpha)
        
        gb_inf_test.fit(self.test_.drop(columns=self.target), self.test_[self.target])
        gb_sup_test.fit(self.test_.drop(columns=self.target), self.test_[self.target])
        
        predictions = pd.DataFrame(self.test_[self.target])
        predictions['lower_gradient_boosting_conformal'] = gb_inf_test.predict(self.test_.drop(columns=self.target)) - qhat
        predictions['upper_gradient_boosting_conformal'] = gb_sup_test.predict(self.test_.drop(columns=self.target)) + qhat
            
        return predictions