from zenml import step
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple
import numpy as np

@step
def train_model_step(df: pd.DataFrame) -> Tuple[LinearRegression, pd.DataFrame, pd.Series]:
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test
