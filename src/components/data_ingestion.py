import pandas as pd
import numpy as np
import os 
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from  dataclasses import dataclass
from src.components.data_transformation import DataTransformation
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class DataIngestionConfig:
    train_path=os.path.join('artifact','train.csv')
    test_path=os.path.join('artifact','test.csv')
    raw_data_path=os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion=DataIngestionConfig()  

    def dataIngestion(self):
        logging.info("Inside data ingestion method")    
        try:
            logging.info("Reading the data..")
            data=pd.read_csv('notebook\data\LoanApprovalData.csv')
            logging.info("Data read successfully")
            os.makedirs(os.path.dirname(self.data_ingestion.train_path),exist_ok=True)
            logging.info("Data saved in the  train path")
            data.to_csv(self.data_ingestion.raw_data_path,index=False)
            data_train,data_test=train_test_split(data)
            data_train=pd.DataFrame(data_train)
            data_test=pd.DataFrame(data_test)
            data_train.to_csv(self.data_ingestion.train_path,index=False)
            data_test.to_csv(self.data_ingestion.test_path,index=False)

            logging.info("Data reading and saving successfull..")
            return (self.data_ingestion.train_path,self.data_ingestion.test_path)    
        except Exception as e:
            raise CustomException("Error while reading data",sys)
            
            
if __name__=='__main__':
    di=DataIngestion()
    train_data_path,test_data_path=di.dataIngestion()
    print(train_data_path,test_data_path)
    data_transformation=DataTransformation()
    data_transformation.initiate_data_tranformation(train_data_path,test_data_path)