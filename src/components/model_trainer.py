import pandas as pd, numpy as np
import os, sys
from src.exceptions import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.utils import evaluate_model

from dataclasses import dataclass

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
        trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
         self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_arr):
          try:
                logging.info("Splitting dependent and independent variables into train and test data")
                X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_arr[:,:-1], test_arr[:,-1])

                models = {
                      "LinearRegression": LinearRegression(),
                      "Lasso": Lasso(),
                      "Ridge": Ridge(),
                      "Elasticnet": ElasticNet(),
                      "DecisionTree": DecisionTreeRegressor()
                }

                report:dict = evaluate_model(X_train, y_train, X_test, y_test, models)

                print(report)
                print('\n====================================================================================\n')
                logging.info(f'Model Report : {report}')

                best_model_score = max(sorted(list(report.values())))  # have a check here
                best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]

                best_model = models[best_model_name]

                print(f"The best model is: {best_model_name} with r2 score {best_model_score} ")
                print("\n***********************************************************************\n")
                logging.info(f"The best model is: {best_model_name} with r2 score {best_model_score} ")

                save_object(file_path=ModelTrainerConfig.trained_model_file_path, obj=best_model)



          except Exception as e:
                logging.info("An exception occurred in model_trainer.py. Error: ",e)
                raise CustomException(e, sys)



