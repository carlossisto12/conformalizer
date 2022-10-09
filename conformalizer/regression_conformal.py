"""regression_conformal module."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm


class ConformalRegression:
    """ConformalRegression class."""

    def __init__(self, df, target, n_calib, random_state=42,
                 test_size=0.2, scaler=None):
        """
        Initialize ConformalRegression class.

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
        self.calib = self.df.sample(n=self.n_calib,
                                    random_state=self.random_state)
        self.df = self.df[~self.df.index.isin(self.calib.index)]
        x = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        self.x_calib = self.calib.drop(self.target, axis=1)
        self.y_calib = self.calib[self.target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,  # NOQA: E501
                                                                                y,  # NOQA: E501
                                                                                test_size=self.test_size,  # NOQA: E501
                                                                                random_state=self.random_state)  # NOQA: E501
        if self.scaler is not None:
            self.x_train = self.scaler.fit_transform(self.x_train)
            self.x_test = self.scaler.transform(self.x_test)
            self.x_calib = self.scaler.transform(self.x_calib)

        self.calib_ = pd.concat([pd.DataFrame(self.x_calib,
                                              columns=self.df.drop(columns=self.target).columns),  # NOQA: E501
                                 pd.DataFrame(self.y_calib)],  # NOQA: E501
                                axis=1)
        self.test_ = pd.concat([pd.DataFrame(self.x_test,
                                             columns=self.df.drop(columns=self.target).columns),
                                pd.DataFrame(self.y_test)],
                               axis=1)

    def _compute_qhat(self, calib_pred, alpha):
        """
        Compute the corrected quantile of conformal scores obtained
        from calibration predictions.
        """
        scores = calib_pred[['t_inf-y', 'y-t_sup']].max(axis=1).to_numpy()
        q = np.ceil((self.n_calib+1)*(1-alpha))/self.n_calib
        qhat = np.quantile(scores, q)
        return qhat

    def linear_conformal_regression(self, formula, alpha):
        """
        Compute Conformal Predictions using a linear model.

        Parameters
        ----------
        formula: string
            Formula to be used in rgression.
        alpha: float [0,1]
            Significance level of conformal intervals.

        Returns
        -------
        predictions: DataFrame
            DataFrame with true value and conformal intervals
        """
        model_inf = smf.quantreg(formula, self.calib_).fit(q=alpha/2)
        model_sup = smf.quantreg(formula, self.calib_).fit(q=1-alpha/2)

        model_inf_test = smf.quantreg(formula, self.test_).fit(q=alpha/2)
        model_sup_test = smf.quantreg(formula, self.test_).fit(q=1-alpha/2)

        calib_pred = pd.DataFrame(self.calib_[self.target])
        calib_pred['t_inf'] = model_inf.predict(
            self.calib_.drop(columns=self.target))
        calib_pred['t_sup'] = model_sup.predict(
            self.calib_.drop(columns=self.target))
        calib_pred['t_inf-y'] = -calib_pred[self.target] + calib_pred['t_inf']
        calib_pred['y-t_sup'] = calib_pred[self.target] - calib_pred['t_sup']

        qhat = self._compute_qhat(calib_pred, alpha)

        predictions = pd.DataFrame(self.test_[self.target])
        predictions['lower_linear_conformal'] = model_inf_test.predict(self.test_.drop(columns=self.target)) - qhat  # NOQA: E501
        predictions['upper_linear_conformal'] = model_sup_test.predict(self.test_.drop(columns=self.target)) + qhat  # NOQA: E501
        return predictions

    def random_forest_conformal_regression(self, alpha, parameters=None):
        """
        Compute Conformal Predictions using RandomForestRegressor.

        Parameters
        ----------
        alpha: float [0,1]
            Significance level of conformal intervals.
        parameters: dict, optional(default=None)
            Parameters to be used in RandomForestRegressor.
            If None, the default values are used.

        Returns
        -------
        predictions: DataFrame
            DataFrame with true value and conformal intervals
        """
        if parameters is not None:
            rf = RandomForestRegressor(**parameters)
            rf_test = RandomForestRegressor(**parameters)
        else:
            rf = RandomForestRegressor()
            rf_test = RandomForestRegressor()
        rf.fit(self.calib_.drop(columns=self.target).values, self.calib_[self.target])
        calib_pred = pd.DataFrame()
        for pred in tqdm(rf.estimators_):
            temp = pd.Series(
                pred.predict(self.calib_.drop(columns=self.target)))
            calib_pred = pd.concat([calib_pred, temp], axis=1)

        calib_pred = pd.DataFrame(self.calib_[self.target])
        calib_pred['t_inf'] = calib_pred.quantile(alpha/2, axis=1)
        calib_pred['t_sup'] = calib_pred.quantile(1-alpha/2, axis=1)
        calib_pred['t_inf-y'] = -calib_pred[self.target] + calib_pred['t_inf']
        calib_pred['y-t_sup'] = calib_pred[self.target] - calib_pred['t_sup']

        qhat = self._compute_qhat(calib_pred, alpha)

        rf_test.fit(self.test_.drop(columns=self.target).values,
                    self.test_[self.target])
        pred_rf_test = pd.DataFrame()
        for pred in tqdm(rf_test.estimators_):
            temp = pd.Series(
                pred.predict(self.test_.drop(columns=self.target)))
            pred_rf_test = pd.concat([pred_rf_test, temp], axis=1)

        predictions = pd.DataFrame(self.test_[self.target])
        predictions['lower_random_forest_conformal'] = pred_rf_test.quantile(alpha/2, axis=1) - qhat  # NOQA: E501
        predictions['upper_random_forest_conformal'] = pred_rf_test.quantile(1-alpha/2, axis=1) + qhat  # NOQA: E501
        return predictions

    def gradient_boosting_conformal_regression(self, alpha, parameters=None):
        """
        Compute Conformal Predictions using GradientBoostingRegressor.

        Parameters
        ----------
        alpha: float [0,1]
            Significance level of conformal intervals.
        parameters: dict, optional(default=None)
            Parameters to be used in GradientBoostingtRegressor.
            If None, the default values are used.

        Returns
        -------
        predictions: DataFrame
            DataFrame with true value and conformal intervals
        """
        if parameters is not None:
            gb_inf = GradientBoostingRegressor(loss='quantile',
                                               alpha=alpha/2, **parameters)
            gb_inf_test = GradientBoostingRegressor(loss='quantile',
                                                    alpha=alpha/2,
                                                    **parameters)
            gb_sup = GradientBoostingRegressor(loss='quantile',
                                               alpha=1-alpha/2, **parameters)
            gb_sup_test = GradientBoostingRegressor(loss='quantile',
                                                    alpha=1-alpha/2,
                                                    **parameters)
        else:
            gb_inf = GradientBoostingRegressor(loss='quantile', alpha=alpha/2)
            gb_inf_test = GradientBoostingRegressor(loss='quantile',
                                                    alpha=alpha/2)
            gb_sup = GradientBoostingRegressor(loss='quantile',
                                               alpha=1-alpha/2)
            gb_sup_test = GradientBoostingRegressor(loss='quantile',
                                                    alpha=1-alpha/2)

        gb_inf.fit(self.calib_.drop(columns=self.target),
                   self.calib_[self.target])
        gb_sup.fit(self.calib_.drop(columns=self.target),
                   self.calib_[self.target])

        calib_pred = pd.DataFrame(self.calib_[self.target])
        calib_pred['t_inf'] = gb_inf.predict(self.calib_.drop(
            columns=self.target))
        calib_pred['t_sup'] = gb_sup.predict(self.calib_.drop(
            columns=self.target))
        calib_pred['t_inf-y'] = -calib_pred[self.target] + calib_pred['t_inf']
        calib_pred['y-t_sup'] = calib_pred[self.target] - calib_pred['t_sup']

        qhat = self._compute_qhat(calib_pred, alpha)

        gb_inf_test.fit(self.test_.drop(columns=self.target),
                        self.test_[self.target])
        gb_sup_test.fit(self.test_.drop(columns=self.target),
                        self.test_[self.target])

        predictions = pd.DataFrame(self.test_[self.target])
        predictions['lower_gradient_boosting_conformal'] = gb_inf_test.predict(self.test_.drop(columns=self.target)) - qhat  # NOQA: E501
        predictions['upper_gradient_boosting_conformal'] = gb_sup_test.predict(self.test_.drop(columns=self.target)) + qhat  # NOQA: E501
        return predictions
