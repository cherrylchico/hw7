from ..interfaces.base import ModelInterface
from typing import Iterable
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class Model(ModelInterface):
    """
    Machine learning model wrapper for classification tasks.
    Supports: LogisticRegression, RandomForestClassifier
    """
    
    def __init__(self, model : str, 
                      feature_cols : Iterable[str], 
                      target_col : str, 
                      **hyperparameters):

        # Private attributes
        self._feature_cols = list(feature_cols)
        self._target_col = target_col
        self._hyperparameters = hyperparameters

        if model == 'logistic_regression':
            self.model = LogisticRegression(**hyperparameters)
        elif model == 'RandomForest':
            self.model = RandomForestClassifier(**hyperparameters)
        else:
            raise ValueError(f"Model {model} not supported.")
    

    def train(self, X : pd.DataFrame, y : pd.Series) -> None:
        X_filtered = X[self._feature_cols]
        self.model.fit(X_filtered, y)
        
    def predict(self, X : pd.DataFrame) -> pd.DataFrame:
        X_filtered = X[self._feature_cols]
        probabilities = self.model.predict_proba(X_filtered)

        return pd.DataFrame(
            probabilities,
            columns=[f"class_{i}" for i in range(probabilities.shape[1])]
        )

    