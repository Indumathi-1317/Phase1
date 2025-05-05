from zenml import step
from sklearn.linear_model import LinearRegression
from typing import Tuple
import pandas as pd
import numpy as np

@step
def train_model(X: pd.DataFrame, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model
