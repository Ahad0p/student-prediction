import os
import sys
import argparse
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
import mlflow
import mlflow.sklearn
import dagshub


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self, learning_rate=0.1, n_estimators=100):
        self.model_trainer_config = ModelTrainerConfig()
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # ✅ Only one model for demonstration (you can expand)
            model = XGBRegressor(
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Evaluate
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)

            # ✅ Initialize MLflow (DagsHub auto-connection)
            dagshub.init(repo_owner='Ahad0p', repo_name='student-prediction', mlflow=True)
            mlflow.set_experiment("XGB_CLI_Param_Experiments")

            with mlflow.start_run(run_name=f"lr={self.learning_rate}_n={self.n_estimators}"):
                mlflow.log_param("learning_rate", self.learning_rate)
                mlflow.log_param("n_estimators", self.n_estimators)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.sklearn.log_model(model, artifact_path="XGB_Model")

            logging.info(f"✅ Run Complete | R²: {r2:.3f} | MAE: {mae:.3f} | MSE: {mse:.3f}")
            save_object(self.model_trainer_config.trained_model_file_path, model)

            return r2

        except Exception as e:
            raise CustomException(e, sys)


# ✅ CLI entry point (this runs only when executed directly)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with dynamic CLI parameters")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for XGBoost")
    parser.add_argument("--n", type=int, default=100, help="Number of estimators")
    args = parser.parse_args()

    # Example: Simulate already-prepared train/test arrays (replace with your data ingestion pipeline)
    import numpy as np
    X_train, X_test = np.random.rand(100, 5), np.random.rand(20, 5)
    y_train, y_test = np.random.rand(100), np.random.rand(20)
    train_array = np.c_[X_train, y_train]
    test_array = np.c_[X_test, y_test]

    trainer = ModelTrainer(learning_rate=args.lr, n_estimators=args.n)
    trainer.initiate_model_trainer(train_array, test_array)
