"""classification_conformal module"""

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from tqdm import tqdm


class ConformalClassification:
    """ConformalClassification class."""

    def __init__(self, df, target, n_calib, random_state=42,
                 test_size=0.2, scaler=None):
        """
        Initialize ConformalClassification class.

        It divides the data in train, test and calibration data; Scale,
        if needed and encode the target column.

        Parameters
        ----------
        df: DataFrame
            Complete DataFrame to be used in the model.
        target: string
            Name of the target column.
        n_calib: int
            Number of observations to be used in calibration.
        random_state: int (optional, default = 42)
            Seed to use in sets division.
        test_size: float [0,1] (optional, default = 0.2)
            Proportion of data to be used in test, excluding calibration set
            size.
        scaler: sklearn Scaler object (optional, default = None)
            Scaler to be used to scale X variables. If None, the values are not
            scaled
        """
        self.df = df
        self.target = target
        self.n_calib = n_calib
        self.random_state = random_state
        self.test_size = test_size
        self.scaler = scaler
        self.dict_map = self._encode_target()
        self.df = self.df.merge(self.dict_map, on=self.target)
        self.calib = self.df.sample(n=self.n_calib,
                                    random_state=self.random_state)
        self.df = self.df[~self.df.index.isin(self.calib.index)]
        x = self.df.drop(self.target, axis=1)
        y = self.df[self.target + '_encode']
        self.x_calib = self.calib.drop(self.target, axis=1)
        self.y_calib = self.calib[self.target + '_encode']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,  # NOQA: E501
                                                                                y,  # NOQA: E501
                                                                                test_size=self.test_size,  # NOQA: E501
                                                                                random_state=self.random_state)  # NOQA: E501
        if self.scaler is not None:
            self.x_train = self.scaler.fit_transform(self.x_train)
            self.x_test = self.scaler.transform(self.x_test)
            self.x_calib = self.scaler.transform(self.x_calib)

    def _encode_target(self):
        """Encode target variable."""
        dict_map = {}
        labels = self.df[self.target].unique()
        labels = np.sort(labels)
        for i in range(0, len(labels)):
            dict_map[labels[i]] = i

        dict_map = pd.DataFrame.from_dict(dict_map,
                                          orient='index').reset_index()
        dict_map.columns = [self.target, self.target+'_encode']
        return dict_map

    def _compute_qhat(self, calib_pred, alpha):
        """
        Compute the corrected quantile of conformal scores obtained
        from calibration predictions.
        """
        q = np.ceil((self.n_calib+1)*(1-alpha))/self.n_calib
        scores = []
        for i in range(calib_pred.shape[0]):
            scores.append(1 - calib_pred[i][self.y_calib.to_numpy()[i]])
        qhat = np.quantile(scores, q)
        return qhat, q

    def _compute_standard_conformal(self, qhat, test_pred):
        """
        Compute non-greedy conformal predictions.
        """
        preds = []
        for i in tqdm(range(test_pred.shape[0])):
            preds_soft = self.dict_map.copy()
            preds_soft['soft_prob'] = test_pred[i]
            preds_soft = preds_soft[preds_soft['soft_prob'] > 1-qhat]
            preds.append(preds_soft[self.target].tolist())
        return preds

    def _compute_adaptive_conformal(self, q, test_pred):
        """
        Compute greedy conformal predictions.
        """
        preds_adaptive = []
        for i in tqdm(range(test_pred.shape[0])):
            preds_soft = self.dict_map.copy()
            preds_soft['soft_prob'] = test_pred[i]
            preds_soft.sort_values(by='soft_prob',
                                   ascending=False, inplace=True)
            preds_soft['preds_soft_cumsum'] = preds_soft['soft_prob'].cumsum()
            preds_soft['exceed_q'] = preds_soft['preds_soft_cumsum'] >= q
            preds_soft['exceed_q_cumsum'] = preds_soft['exceed_q'].cumsum()
            preds_soft = preds_soft[preds_soft['exceed_q_cumsum'] <= 1]
            preds_adaptive.append(preds_soft[self.target].tolist())
        return preds_adaptive

    def xgb_conformal_classification(self, alpha, parameters=None,
                                     num_boost_round=30, verbose_eval=True,
                                     method='standard'):
        """
        Compute conformal predictions for a XGBoost model.

        Parameters
        ----------
        alpha : float
            Significance level of classification sets.
        parameters : dict, optional (default: None)
            Parameters to be used in XGBoost model.
        num_boost_round : int, optional (default = 30)
            Boosting rounds in XGBoost training.
        verbose_eval : bool, optional (default = True)
            Wheter to get verbose in XGBoost training.
        method : string, optional(default = 'standard')
            Conformal Prediction method to be used.
            If 'standard', the non-greedy is applied.
            If 'adpative', the greedy is applied.

        Returns
        -------
        conformal : DataFrame
            DataFrame with True label and conformal prediction set.
        """
        if parameters is None:
            parameters = {'objective': 'multi:softprob',
                          'num_class': self.y_train.nunique()}
        xgb_train = xgb.DMatrix(self.x_train, self.y_train.to_numpy())
        xgb_test = xgb.DMatrix(self.x_test, self.y_test)
        xgb_calib = xgb.DMatrix(self.x_calib, self.y_calib)
        xgb_model = xgb.train(parameters, xgb_train,
                              num_boost_round=num_boost_round,
                              verbose_eval=verbose_eval)
        calib_pred = xgb_model.predict(xgb_calib)
        test_pred = xgb_model.predict(xgb_test)
        qhat, q = self._compute_qhat(calib_pred, alpha)

        if method == 'standard':
            conformal_prediction = self._compute_standard_conformal(qhat,
                                                                    test_pred)
        else:
            conformal_prediction = self._compute_adaptive_conformal(q,
                                                                    test_pred)

        conformal = pd.DataFrame()
        conformal[self.target+'_encode'] = self.y_test
        conformal = conformal.merge(self.dict_map, on=self.target+'_encode')
        conformal.drop(columns=self.target+'_encode', inplace=True)
        conformal['conformal_prediction'] = conformal_prediction
        return conformal

    def conformal_classification(self, alpha, classifier, method='standard'):
        """
        Compute conformal predictions for a sklearn classifier.

        Parameters
        ----------
        alpha : float
            Significance level of classification sets.
        classifier : skleran classifier.
            Classifier to be used to build the conformal predictions. It must
            have the predict_proba method.
        method : string, optional(default = 'standard')
            Conformal Prediction method to be used.
            If 'standard', the non-greedy is applied.
            If 'adpative', the greedy is applied.

        Returns
        -------
        conformal : DataFrame
            DataFrame with True label and conformal prediction set.
        """
        classifier.fit(self.x_train, self.y_train)
        calib_pred = classifier.predict_proba(self.x_calib)
        test_pred = classifier.predict_proba(self.x_test)
        qhat, q = self._compute_qhat(calib_pred, alpha)
        if method == 'standard':
            conformal_prediction = self._compute_standard_conformal(qhat,
                                                                    test_pred)
        else:
            conformal_prediction = self._compute_adaptive_conformal(q,
                                                                    test_pred)
        conformal = pd.DataFrame()
        conformal[self.target+'_encode'] = self.y_test
        conformal = conformal.merge(self.dict_map,
                                    on=self.target+'_encode')
        conformal.drop(columns=self.target+'_encode', inplace=True)
        conformal['conformal_prediction'] = conformal_prediction
        return conformal
