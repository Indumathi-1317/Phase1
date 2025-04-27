from zenml import step
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

@step
def evaluate_model_step(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Mean Squared Error: {mse}")
    return mse
