
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
import mlflow
import time # Import time module for sleep
import subprocess # Import subprocess for mlflow ui

# Start MLflow UI on port 5000
process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])
time.sleep(10) # Give MLflow UI time to start

# Set the tracking URL for MLflow to localhost as requested
mlflow.set_tracking_uri("http://localhost:5000")

# Set the name for the experiment
mlflow.set_experiment("MLOps_experiment")

# Load the training and testing data
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

print("Data loaded successfully.")
print(f"Xtrain shape: {Xtrain.shape}, ytrain shape: {ytrain.shape}")
print(f"Xtest shape: {Xtest.shape}, ytest shape: {ytest.shape}")

# Define numerical and categorical features
numerical_features = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome', 'PitchSatisfactionScore'
]
categorical_features = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'MaritalStatus',
    'ProductPitched', 'Designation'
]

# Handle class imbalance# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features), 
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)


param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    classification_threshold = 0.55

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    print(classification_report(ytest, y_pred_test))

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    # Save next to app.py so the Streamlit app can load it directly, and log
    # it as an MLflow artifact for traceability
    model_path = "tourism_project/deployment/best_tourism_project_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
