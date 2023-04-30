from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
import os 
import sys
import dill 
def save_obj(file_path,obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj) 
    except Exception as e:
        raise CustomException(e,sys)
           
def evaluate_model(x_train,y_train,x_test,y_test,models:dict):
    try:
        model_report ={}
        for i in range(len(models)):
            model=list(models.values())[i]
            logging.info("Model training started..")
            model.fit(x_train,y_train)
            logging.info("Model prediction and report generation started")
            y_pred= model.predict(x_test)
            model_report[list(models.values())[i]]=r2_score(y_test,y_pred)
        return model_report         
    except Exception as e:
        raise CustomException(e,sys)    
def load_model(file_path):
    try:
        with open(file_path,"rb") as f:
            obj= dill.load(f)
            return obj

    except Exception as e:
        raise CustomException(e,sys)              