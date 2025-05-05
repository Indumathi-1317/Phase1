from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    df = df.dropna()
    X = df.drop("target", axis=1)
    y = df["target"].values  # Convert to NumPy array
    return X, y
