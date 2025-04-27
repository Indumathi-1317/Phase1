from zenml import pipeline
from steps.load_data import load_data_step
from steps.clean_data import clean_data_step
from steps.train_model import train_model_step
from steps.evaluate_model import evaluate_model_step

@pipeline
def my_pipeline():
    data = load_data_step()
    cleaned = clean_data_step(data)
    model, X_test, y_test = train_model_step(cleaned)
    evaluate_model_step(model, X_test, y_test)

if __name__ == "__main__":
    my_pipeline()  # ✅ Corrected
