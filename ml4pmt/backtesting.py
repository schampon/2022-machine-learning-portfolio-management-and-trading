from sklearn.utils.metaestimators import _safe_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, clone
from ml4pmt.metrics import sharpe_ratio
from joblib import Parallel, delayed
import pandas as pd
import numpy as np


def compute_batch_holdings(pred, V, A=None, past_h=None, constant_risk=False):
    """
    compute markowitz holdings with return prediction "mu" and covariance matrix "V"

    mu: numpy array (shape N * K)
    V: numpy array (N * N)

    """

    N, _ = V.shape
    if isinstance(pred, pd.Series) | isinstance(pred, pd.DataFrame):
        pred = pred.values
    if pred.shape == (N,):
        pred = pred[:, None]
    elif pred.shape[1] == N:
        pred = pred.T

    invV = np.linalg.inv(V)
    if A is None:
        M = invV
    else:
        U = invV.dot(A)
        if A.ndim == 1:
            M = invV - np.outer(U, U.T) / U.dot(A)
        else:
            M = invV - U.dot(np.linalg.inv(U.T.dot(A)).dot(U.T))
    h = M.dot(pred)
    if constant_risk:
        h = h / np.sqrt(np.diag(h.T.dot(V.dot(h))))
    return h.T


class MeanVariance(BaseEstimator):
    def __init__(self, transform_V=None, A=None, constant_risk=True):

        if transform_V is None:
            self.transform_V = lambda x: np.cov(x.T)
        else:
            self.transform_V = transform_V
        self.A = A
        self.constant_risk = constant_risk

    def fit(self, X, y=None):
        self.V_ = self.transform_V(y)

    def predict(self, X):
        if self.A is None:
            T, N = X.shape
            A = np.ones(N)
        else:
            A = self.A
        h = compute_batch_holdings(
            X, self.V_, A, constant_risk=self.constant_risk)
        return h

    def score(self, X, y):
        return sharpe_ratio(np.sum(X * y, axis=1))


class Backtester:
    def __init__(
        self,
        estimator,
        ret,
        max_train_size=36,
        test_size=1,
        start_date="1945-01-01",
        end_date=None,
    ):

        self.start_date = start_date
        self.end_date = end_date
        self.estimator = estimator
        self.ret = ret[: self.end_date]
        self.cv = TimeSeriesSplit(
            max_train_size=max_train_size,
            test_size=test_size,
            n_splits=1 + len(ret.loc[start_date:end_date]) // test_size,
        )

    def train(self, features, target):
        pred, estimators = fit_predict(
            self.estimator, features, target, self.ret, self.cv, return_estimator=True
        )
        self.estimators_ = estimators
        self.h_ = pred
        self.pnl_ = (
            pred.shift(1).mul(self.ret).sum(axis=1)[
                self.start_date: self.end_date]
        )
        return self


def _fit_predict(estimator, X, y, train, test, return_estimator=False):
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)
    estimator.fit(X_train, y_train)
    if return_estimator:
        return estimator.predict(X_test), estimator
    else:
        return estimator.predict(X_test)


def fit_predict(
    estimator,
    features,
    target,
    ret,
    cv,
    return_estimator=False,
    verbose=0,
    pre_dispatch="2*n_jobs",
    n_jobs=1,
):
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    res = parallel(
        delayed(_fit_predict)(
            clone(estimator), features, target, train, test, return_estimator
        )
        for train, test in cv.split(ret)
    )
    if return_estimator:
        pred, estimators = zip(*res)
    else:
        pred = res
    cols = ret.columns
    idx = ret.index[np.concatenate([test for _, test in cv.split(ret)])]
    if return_estimator:
        return pd.DataFrame(np.concatenate(pred), index=idx, columns=cols), estimators
    else:
        return pd.DataFrame(np.concatenate(pred), index=idx, columns=cols)
