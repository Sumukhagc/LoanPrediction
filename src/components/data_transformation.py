import os 
import sys 
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from src.utils import save_obj
@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.preprocessor_config=DataTransformationConfig()

    def preprocessor_data(self):
        try:
            data=pd.read_csv('artifact/train.csv')
            numerical_features=[col for col in data.columns if data[col].dtype!='O']
            categorical_features=[col for col in data.columns if data[col].dtype=='O']
            print("Numerical columns",numerical_features)
            print("categorical columns",categorical_features)
            numerical_pipeline=Pipeline(
                steps=[
                ("impute",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            categorical_pipeline=Pipeline(
                steps=[
                ("impute",SimpleImputer(strategy='most_frequent')),
                ("encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor=ColumnTransformer(transformers=[
                ("num",numerical_pipeline,numerical_features),
                ("cat",categorical_pipeline,categorical_features)
            ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            logging.info("Train and test data read successfully..")
            preprocessor=self.preprocessor_data()
            target_var='GrAppv'
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path) 
            logging.info("Preprocessing on Training and Testing data started... ")
            train_X_transformed=preprocessor.fit_transform(train)
            test_X_transformed= preprocessor.fit_transform(test)
            logging.info("Preprocessing on Training and Testing data is completed... ")
            
            save_obj(
                file_path=self.preprocessor_config.preprocessor_file_path,
                obj=preprocessor
            )
            logging.info('Saved preprecessor pickle file')
        except Exception as e:
            raise CustomException(e,sys)        