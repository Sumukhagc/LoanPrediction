import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.utils import save_obj,evaluate_model
from sklearn.ensemble import AdaBoostRegressor
    
@dataclass
class ModelTrainerConfig:
    model_train_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_train_path=ModelTrainerConfig()
    def train_model(self,x_train,y_train,x_test,y_test):
        try:
            logging.info("Model training has started.")
            models={
                "LinearRegressor":LinearRegression(),
                "DecesionTreeRegressor":DecisionTreeRegressor()
                }
            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)   
            model_report_sorted=dict(sorted(model_report.items(),key=lambda x : x[1],reverse=True))

            logging.info(model_report) 
            logging.info("saving the model")
            best_model=list(model_report_sorted.keys())[0]
            save_obj(ModelTrainerConfig.model_train_path,obj=best_model)
        except Exception as e:
            raise CustomException(e,sys)