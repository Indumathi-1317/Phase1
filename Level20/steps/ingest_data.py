from zenml import step
from sklearn.datasets import load_diabetes
import pandas as pd

@step
def ingest_data() -> pd.DataFrame:
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df
