from zenml import step
from sklearn.datasets import load_digits
import pandas as pd

@step
def load_data_step() -> pd.DataFrame:
    digits = load_digits(as_frame=True)
    df = digits.frame
    return df
