from django.http import HttpResponse
from django.shortcuts import render
from src.logger import logging
from src.exception import CustomException
from src.utils import load_model
import pandas as pd
def home(request):
    return render(request,"home.html")

def predict(request):
    train=pd.read_csv('artifact/X_train.csv')
    preprocessor_path='artifact/preprocessor.pkl'
    model_path='artifact/model.pkl'
    preprocessor=load_model(preprocessor_path)
    input_values=request.POST.dict()
    input_dict={}
    input_dict['NAICS']=[int(input_values['naics'])]
    input_dict['Term']=[int(input_values['term'])]
    input_dict['NoEmp']=[int(input_values['noemp'])]
    input_dict['NewExist']=[float(input_values['age'])]
    input_dict['CreateJob']=[int(input_values['nojob'])]
    input_dict['RetainedJob']=[int(input_values['noretained'])]
    input_dict['UrbanRural']=[float(input_values['urbanrural'])]
    input_dict['RevLineCr']=[(input_values['revlinecr'])]
    input_dict['LowDoc']=[(input_values['lowdoc'])]
    input_dict['DisbursementGross']=[int(input_values['disbursment'])]
    input_dict['BalanceGross']=[int(input_values['balance'])]
    input_dict['MIS_Status']=[(input_values['mis'])]
    input_dict['ChgOffPrinGr']=[float(input_values['chgoff'])]
    input_df=pd.DataFrame(input_dict)
    input_transformed=preprocessor.transform(input_df)
    model=load_model(model_path)
    pred=model.predict(input_transformed)
    ans={"pred":pred[0]}
    return render(request,"predict.html",ans)