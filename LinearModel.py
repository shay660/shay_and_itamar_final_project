from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression


class RNALinearModel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, max_iter=2500, tol=5e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.lasso = Lasso(max_iter=max_iter, tol=tol, alpha=alpha)
        self.linear_reg = LinearRegression()
        self.fitted = False


    def fit(self, X, y):
        self.lasso.fit(X, y)
        self.filterd_feat = (self.lasso.coef_ != 0).any(axis=0)
        X = X.loc[:, self.filterd_feat]
        self.linear_reg.fit(X, y)
        self.coef_ = self.linear_reg.coef_
        self.feature_names_in_ = self.linear_reg.feature_names_in_
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Model should be fitted before prediction.\n")
        X = X.loc[:, self.filterd_feat]
        return self.linear_reg.predict(X)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score
        return r2_score(y_true=y, y_pred=self.predict(X))

    def set_params(self, **params):
        # Update parameters dynamically and ensure they propagate to Lasso
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)

        # Update Lasso after parameter changes
        self.lasso = Lasso(max_iter=self.max_iter, tol=self.tol,
                           alpha=self.alpha)
        return self

    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol
        }