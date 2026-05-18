import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("MLOps_Project")

client = MlflowClient()

df = pd.read_csv("data/housing_v1.csv")

X = df.drop("target", axis=1)
y = df["target"]

models = {
    "LR": LinearRegression(),
    "RF": RandomForestRegressor(),
    "GB": GradientBoostingRegressor()
}

best_rmse = float("inf")
best_run = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X, y)
        preds = model.predict(X)

        rmse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)

        mlflow.log_param("model", name)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, "model")

        if rmse < best_rmse:
            best_rmse = rmse
            best_run = mlflow.active_run().info.run_id

# REGISTER MODEL
model_uri = f"runs:/{best_run}/model"
result = mlflow.register_model(model_uri, "HousingModel")

# STAGING
client.transition_model_version_stage(
    name="HousingModel",
    version=result.version,
    stage="Staging"
)

# PRODUCTION
if best_rmse < 50:
    client.transition_model_version_stage(
        name="HousingModel",
        version=result.version,
        stage="Production"
    )
# test
