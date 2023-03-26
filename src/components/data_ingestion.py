import pandas as pd
import numpy as np
import os 
import sys
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from  dataclasses import dataclass

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
            data=pd.read_csv('notebook\data\credit_train.csv')
            logging.info("Data read successfully")
            os.makedirs(os.path.dirname(self.data_ingestion.train_path),exist_ok=True)
            logging.info("Data saved in the  train path")
            data.to_csv(self.data_ingestion.raw_data_path)
            data_train=data=pd.read_csv('notebook\data\credit_train.csv')
            data_test=data=pd.read_csv('notebook\data\credit_test.csv')
            data_train.to_csv(self.data_ingestion.train_path)
            data_test.to_csv(self.data_ingestion.test_path)
            logging.info("Data reading and saving successfull..")
            return (data_train,data_test)    
        except Exception as e:
            raise CustomException("Error while reading data",sys)
            print("gc",e)
            pass
if __name__=='__main__':
    di=DataIngestion()
    di.dataIngestion()