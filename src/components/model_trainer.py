import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModuleTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pk1")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModuleTrainerConfig()

    def initiate_model_trainer(self,train_array,test_arry):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_arry[:,:-1],
                test_arry[:,-1]
            )

            models= {

            "Randomforest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Classifier": KNeighborsRegressor(),
            "XGBClassifier":XGBRegressor(),
            "CatBoostomg classifier": CatBoostRegressor(),
            "AdaBoosting Classifier": AdaBoostRegressor(),
            }
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_mode_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_mode_score)
            ]

            best_model = models[best_model_name]

            if best_mode_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )
            
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
            



