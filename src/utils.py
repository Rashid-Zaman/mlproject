import os
import sys

import pandas as pd
import numpy as np
import dill

from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        r2_report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            
            r2_report[list(models.keys())[i]]=r2_score(y_test,y_pred)
        return r2_report
    except Exception as e:
        raise CustomException(e,sys)
        
        
    