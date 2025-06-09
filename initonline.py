from sklearn.datasets import load_iris
import pandas as pd
import mlflow
from sklearn.linear_model import SGDClassifier
from joblib import dump

# Load the iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Initialize MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Online Training Iris Experiment")

# Model Online Learning: SGDClassifier
model = SGDClassifier(loss='log_loss', learning_rate='adaptive', max_iter=10000, eta0=0.01)

# Kelas target (diperlukan untuk partial_fit)
classes = data['target'].unique()

with mlflow.start_run():
    mlflow.autolog()  # Enable autologging

    # Preprocessing data untuk batch
    X_batch = data.drop(columns=['target'])
    y_batch = data['target']

    # Initial fit untuk first batch
    model.partial_fit(X_batch, y_batch, classes=classes)

    # Log metrik after setiap batch
    accuracy = model.score(X_batch, y_batch)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")
    # Save the model
    dump(model, 'sgd_classifier_iris.joblib')

    # Log file model sebagai artifact to mlflow
    mlflow.log_artifact('sgd_classifier_iris.joblib', artifact_path='model_artifacts')

    # Log model after training
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="online_model",
        input_example=X_batch.iloc[:5]
    )

