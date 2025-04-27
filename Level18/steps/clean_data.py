from zenml import step
import pandas as pd

@step
def clean_data_step(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns if needed (e.g., drop 'images' if present)
    if 'images' in df.columns:
        df = df.drop(columns=['images'])
    
    # Fill NA values
    df = df.fillna(df.median(numeric_only=True))
    return df
