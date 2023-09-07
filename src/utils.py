import os, sys, pickle
from src.exceptions import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_object( file_path, obj):
    try:
        
        dir_path = os.path.dirname(file_path)

        if not os.path.exists(dir_path):
            logging.info("Creating directory: ",dir_path)
        os.makedirs(dir_path, exist_ok= True)

        print(file_path,str(obj))

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            test_module_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_module_score

        return report


    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)