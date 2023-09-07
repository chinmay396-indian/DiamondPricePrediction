import os, sys
from src.logger import logging
from src.exceptions import CustomException
from src.utils import load_object
import pandas as pd, numpy as np

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, input_data):

        try:
            logging.info("Prediction started")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            preprocessor = load_object(preprocessor_path)
            logging.info("Processor created and loaded.")
            model_path = os.path.join("artifacts","model.pkl")
            model = load_object(model_path)
            logging.info("Model created and loaded")
            logging.info(type(input_data),preprocessor)
            scaled_data = preprocessor.transform(input_data)
            logging.info("input_data scaled")
            print("prediction1")
            print(scaled_data)

            pred = model.predict(scaled_data)


            return pred
        
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                carat:float,
                depth:float,
                table:float,
                x:float,
                y:float,
                z:float,
                cut:str,
                color:str,
                clarity:str) :
        
        self.carat = carat,
        self.depth = depth,
        self.table = table,
        self.x = x,
        self.y = y,
        self.z = z,
        self.cut = cut,
        self.color = color,
        self.clarity = clarity

    def get_data_as_dataframe(self):
        
        try:
            custom_data_input_dict = {
                'carat':self.carat,
                'depth':self.depth,
                'table':self.table,
                'x':self.x,
                'y':self.y,
                'z':self.z,
                'cut':self.cut,
                'color':self.color,
                'clarity':[self.clarity]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Converted input data into DataFrame.")
            return df


        except Exception as e:
            logging.info("An Exception occurred !")
            raise CustomException(e,sys)


    





            

