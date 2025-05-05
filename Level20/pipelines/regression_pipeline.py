from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

@pipeline
def regression_pipeline():
    df = ingest_data()
    X, y = clean_data(df)
    model = train_model(X, y)
    evaluate_model(model, X, y)
