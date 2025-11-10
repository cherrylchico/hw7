from abc import ABC, abstractmethod
from typing import Tuple, Iterable
import pandas as pd



# DATA LOADING INTERFACE

class DataLoaderInterface(ABC):
    """Interface for loading and splitting data"""
    
    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data and split into train and test sets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        raise NotImplementedError("Subclasses must implement load method")


# PREPROCESSING INTERFACE

class PreprocessorInterface(ABC):
    """Interface for data preprocessing steps"""
    
    @abstractmethod
    def fit(self, df : pd.DataFrame | None = None, cols : Iterable[str] | None = None) -> 'PreprocessorInterface':
        """
        Fit the preprocessor to the data.
        
        Args:
            df: DataFrame to fit on (None if not needed)
            cols: List of columns to consider (None if not needed)
        Returns:
            self: Returns self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit method")
    
    @abstractmethod
    def transform(self, df: pd.DataFrame, cols : Iterable[str]  |  None = None) -> pd.DataFrame:
        """
        Transform the data according to preprocessing logic.
        
        Args:
            df: DataFrame to transform
            cols: List of columns to consider
        Returns:
            pd.DataFrame: Transformed data
        """
        raise NotImplementedError("Subclasses must implement transform method")
    
    def fit_transform(self, df: pd.DataFrame, cols : Iterable[str]  |  None = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            cols: List of columns to consider
        Returns:
            pd.DataFrame: Transformed data
        """
        self.fit(df, cols)
        return self.transform(df, cols)


# FEATURE ENGINEERING INTERFACE

class FeatureEngineerInterface(ABC):
    """
    Contract for feature engineers.
    Implementations should be able to fit to a DataFrame (optional) and transform it.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, target: pd.Series | None = None) -> "FeatureEngineerInterface":
        """Learn any statistics or encodings from df. Return self."""
        raise NotImplementedError("Subclasses must implement fit method")

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to df and return transformed DataFrame."""
        raise NotImplementedError("Subclasses must implement transform method")

    def fit_transform(self, df: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
        """Default fit_transform that calls fit then transform."""
        self.fit(df, target=target)
        return self.transform(df)
    

# MODELING INTERFACE

class ModelInterface(ABC):
    """Interface for machine learning models"""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on the provided features and target.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict probabilities for the given features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            pd.DataFrame: Predicted probabilities
        """
        raise NotImplementedError("Subclasses must implement predict method")