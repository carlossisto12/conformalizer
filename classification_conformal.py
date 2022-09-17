import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
import os


class ConformalClassification:
    def __init__(self, df, target, n_calib, random_state=42, test_size=0.2, scaler=None):
        self.df = df
        self.target = target
        self.n_calib = n_calib
        self.random_state = random_state
        self.test_size = test_size
        self.scaler = scaler
        self.dict_map = self._encode_target()
        print(self.dict_map)
        self.df = self.df.merge(self.dict_map, on = self.target)
        self.calib = self.df.sample(n=self.n_calib, random_state = self.random_state)
        self.df = self.df[~self.df.index.isin(self.calib.index)]
        x = self.df.drop(self.target, axis=1)
        y = self.df[self.target + '_encode']
        self.x_calib = self.calib.drop(self.target, axis=1)
        self.y_calib = self.calib[self.target + '_encode']
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
        
        
        
    def _encode_target(self):
        dict_map = {}
        labels = self.df[self.target].unique()
        for i in range(0,len(labels)):
            dict_map[labels[i]] = i
            
        dict_map = pd.DataFrame.from_dict(dict_map, orient='index').reset_index()
        dict_map.columns = [self.target, self.target+'_encode']
        return dict_map
    
    def _compute_qhat(self, calib_pred, alpha):
        q = np.ceil((self.n_calib+1)*(1-alpha))/self.n_calib
        scores  = []
        for i in range(calib_pred.shape[0]):
            scores.append(1 - calib_pred[i][self.y_calib.to_numpy()[i]])
        qhat = np.quantile(scores, q)
        return qhat, q
    
    def _compute_standard_conformal(self, qhat, test_pred):
        preds = []
        for i in tqdm(range(test_pred.shape[0])):
            preds_soft = self.dict_map.copy()
            preds_soft['soft_prob'] = test_pred[i]
            preds_soft = preds_soft[preds_soft['soft_prob']>1-qhat]
            preds.append(preds_soft[self.target].tolist())
        return preds
    
    def _compute_adaptive_conformal(self, q, test_pred):
        preds_adaptive = []
        for i in tqdm(range(test_pred.shape[0])):
            preds_soft = self.dict_map.copy()
            preds_soft['soft_prob'] = test_pred[i]
            preds_soft.sort_values(by = 'soft_prob', ascending=False, inplace=True)
            preds_soft['preds_soft_cumsum'] = preds_soft['soft_prob'].cumsum()
            preds_soft['exceed_q'] = preds_soft['preds_soft_cumsum'] >= q
            preds_soft['exceed_q_cumsum'] = preds_soft['exceed_q'].cumsum()
            preds_soft = preds_soft[preds_soft['exceed_q_cumsum'] <= 1]
            preds_adaptive.append(preds_soft[self.target].tolist())
        return preds_adaptive
        
            
        
    def xgb_conformal_classification(self,alpha, parameters=None, num_boost_round=30, verbose_eval=True, method='standard'):
        if parameters is None:
            parameters = {'objective': 'multi:softprob',
                          'num_class': self.y_train.nunique()}
        xgb_train = xgb.DMatrix(self.x_train, self.y_train.to_numpy())
        xgb_test = xgb.DMatrix(self.x_test, self.y_test)
        xgb_calib = xgb.DMatrix(self.x_calib, self.y_calib)
        xgb_model = xgb.train(parameters, xgb_train, num_boost_round=num_boost_round, verbose_eval=verbose_eval)
        calib_pred = xgb_model.predict(xgb_calib)
        test_pred = xgb_model.predict(xgb_test)
        qhat, q = self._compute_qhat(calib_pred, alpha)
        
        if method == 'standard':
            conformal_prediction = self._compute_standard_conformal(qhat, test_pred)
        else:
            conformal_prediction = self._compute_adaptive_conformal(q, test_pred)
            
        conformal = pd.DataFrame()
        conformal[self.target+'_encode'] = self.y_test
        conformal = conformal.merge(self.dict_map, on = self.target+'_encode')
        conformal.drop(columns=self.target+'_encode', inplace=True)
        conformal['conformal_prediction'] = conformal_prediction
        return conformal
        
        