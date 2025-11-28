import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import mlflow
import mlflow.sklearn
import dagshub  # ✅ DagsHub integration


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
            }

            # Evaluate all models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
            best_model = models[best_model_name]

            # Train best model
            best_model.fit(X_train, y_train)

            # Predictions and metrics
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)

            # ✅ Initialize DagsHub MLflow integration
            dagshub.init(repo_owner='Ahad0p', repo_name='student-prediction', mlflow=True)

            # ✅ Start MLflow run
            with mlflow.start_run(run_name=f"{best_model_name}_run"):
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_metric("r2_score", r2_square)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.sklearn.log_model(best_model, artifact_path=best_model_name)

            # Raise error if model is too weak
            if best_model_score < 0.6:
                raise CustomException("No suitable best model found (score < 0.6)")

            logging.info(
                f"Best Model: {best_model_name} | R²: {r2_square:.3f} | MAE: {mae:.3f} | MSE: {mse:.3f}"
            )

            # Save best model locally
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
