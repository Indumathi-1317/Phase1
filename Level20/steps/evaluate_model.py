from zenml import step
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

@step
def evaluate_model(model: LinearRegression, X: pd.DataFrame, y: np.ndarray) -> None:
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"[evaluate_model] MSE: {mse}")
    print(f"[evaluate_model] R2 Score: {r2}")
