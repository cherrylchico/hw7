#Create a class with a primary method that loads the data and returns two dataframes, one for
#train and another for test. Internally, the class can use the function defined in hw5.
import pandas as pd
from ..interfaces.base import DataLoaderInterface
from typing import Tuple

class DataLoader(DataLoaderInterface):

    def __init__(self, filepath : str, frac : float, random_state : int):
        self.filepath = filepath
        self.frac = frac
        self.random_state = random_state

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data and split into train and test sets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        df = pd.read_csv(self.filepath, index_col=0)
        train_df, test_df = self._split_data(df)
        return train_df, test_df
    
    def _split_data(self, df : pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_df = df.sample(frac=self.frac, random_state=self.random_state)
        train_df = df.drop(index=test_df.index)
        return train_df, test_df
        
