import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

# 1. Cargar el dataset de Wine
data = load_wine()
X = data.data
y = data.target

# 2. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = 0.2, 
    random_state = 42
)

# Configuración de MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("wine-quality-experiment")

with mlflow.start_run():
    # Train a model
    random_state = 2
    model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    mlflow.log_param("random_state", random_state)
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log metrics and model
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")