import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

from sklearn.metrics import r2_score
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                "RandomForest":RandomForestRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "GradientBoost":GradientBoostingRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "Linear Regression":LinearRegression(),
                "xgboost":XGBRegressor(),
                "catboost":CatBoostRegressor(),
                "kneighbours":KNeighborsRegressor()
            }
            
            model_report:dict=evaluate_model(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models)
            
            ## to get best_model_score
            best_model_score=max(model_report.values())
            
            ## to get the best model
            best_model=list(models.values())[list(model_report.values()).index(best_model_score)]
            
            if best_model_score < 0.6:
                raise CustomException("No best Model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            score=r2_score(y_test,predicted)
            
            return score
            
            
        except Exception as e:
            raise CustomException(e,sys)
