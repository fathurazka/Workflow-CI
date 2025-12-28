import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import numpy as np
import os
import warnings
import sys

from automate_fathurazka_modelling import preprocess_data

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    

    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.csv')
    data = pd.read_csv(file_path)
    
    X_train, X_test, y_train, y_test = preprocess_data(data=data, target_column='fraud', save_path="preprocessor.joblib", file_path="data_columns.csv")
    
    input_example = X_train[0:5]
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_tracking_uri("file:./mlruns")
    #mlflow.set_experiment("Fraud_Detection")
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        """
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            input_example=input_example
        )
        
        
        # Get accuracy, precision, recall, f1-score
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        """