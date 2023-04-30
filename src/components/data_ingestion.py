import pandas as pd
import numpy as np
import os 
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.components.model_prediction import ModelTrainer
from  dataclasses import dataclass
from src.components.data_transformation import DataTransformation
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class DataIngestionConfig:
    X_train_path=os.path.join('artifact','X_train.csv')
    X_test_path=os.path.join('artifact','X_test.csv')
    Y_train_path=os.path.join('artifact','Y_train.csv')
    Y_test_path=os.path.join('artifact','Y_test.csv')
    raw_data_path=os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion=DataIngestionConfig()  

    def dataIngestion(self):
        logging.info("Inside data ingestion method")    
        try:
            logging.info("Reading the data..")
            data=pd.read_csv('notebook\data\LoanApprovalData.csv')
            data.drop('SBA_Appv',inplace=True,axis=1)
            data=data[data['MIS_Status']=='P I F']
            data=data.iloc[:,1:]
            logging.info("Data read successfully")
            #os.makedirs(os.path.dirname(self.data_ingestion.X_train_path),exist_ok=True)
            logging.info("Data saved in the train path")
            data.to_csv(self.data_ingestion.raw_data_path,index=False)
            data_X=data.drop(['GrAppv'],axis=1)
            data_Y=data['GrAppv']
            data_X_train,data_X_test,data_Y_train,data_Y_test=train_test_split(data_X,data_Y,test_size=0.2,random_state=33)
            #data_train,data_test=train_test_split(data)
            #data_train=pd.DataFrame(data_train)
            #data_test=pd.DataFrame(data_test)
            data_X_train.to_csv(self.data_ingestion.X_train_path,index=False)
            data_X_test.to_csv(self.data_ingestion.X_test_path,index=False)
            data_Y_train.to_csv(self.data_ingestion.Y_train_path,index=False)
            data_Y_test.to_csv(self.data_ingestion.Y_test_path,index=False)
            #data_test.to_csv(self.data_ingestion.test_path,index=False)

            logging.info("Data reading and saving successfull..")
            return (data_X_train,data_X_test,data_Y_train,data_Y_test)    
        except Exception as e:
            raise CustomException("Error while reading data",sys)
            
 ##This code is to test the functionality           
if __name__=='__main__':
    di=DataIngestion()
    di.dataIngestion()
    data_transformation=DataTransformation()
    x_train,x_test,y_train,y_test=data_transformation.initiate_data_tranformation()
    model=ModelTrainer()
    model.train_model(x_train,y_train,x_test,y_test)