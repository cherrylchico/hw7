#Create a preprocessor class that removes those rows that contain NaN values in the columns: age,
#gender, ethnicity.
#c. Create a preprocessor class that fills NaN with the mean value of the column in the columns:
#height, weight.

import pandas as pd
from typing import Iterable
from ..interfaces.base import PreprocessorInterface

class DropNA(PreprocessorInterface):
    """
    Dataframe preprocessor class 
    Specifically for dropping null values of specified columns
    """
    def __init__(self):
        pass

    def fit(self, df : pd.DataFrame | None = None, cols : Iterable[str] | None = None) -> 'DropNA':
        return self

    def transform(self, df, cols : Iterable[str] |  None = None) -> pd.DataFrame:
        df = df.dropna(subset=cols)
        return df


class ImputeNA(PreprocessorInterface):
    """
    Class that imputes values according to the specified method
    for the specified columns

    The values used for imputation is stored as a class attribute
    """
    def __init__(self, method : str = "mean"):

        self.method = method
        self.impute_reference = {}
    
    def fit(self, df : pd.DataFrame, cols : Iterable[str] |  None = None) -> 'ImputeNA':
        self._impute_map(df, cols)
        return self

    def transform(self, df : pd.DataFrame, cols : Iterable[str] |  None = None) -> pd.DataFrame:

        df = df.copy()
        for col, fill_value in self.impute_reference.items():
            df[col]=df[col].fillna(fill_value)
            
        return df
    

    def _impute_map(self, df : pd.DataFrame, cols : Iterable[str]) -> None:
        for col in cols:
            if self.method == "mean":
                self.impute_reference[col] = self._mean(col, df)
            elif self.method == "median":
                self.impute_reference[col] = self._median(col, df)
            elif self.method == "mode":
                self.impute_reference[col] = self._mode(col, df)
            else:
                raise ValueError(f"Imputation method {self.method} not recognized.")


    def _mean(self, col : Iterable[str], df : pd.DataFrame) -> float:
        return df[col].mean()
    
    def _median(self, col : Iterable[str], df : pd.DataFrame) -> float:
        return df[col].median().iloc[0]
    
    def _mode(self, col : Iterable[str], df : pd.DataFrame):
        return df[col].mode().iloc[0]
