from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
#pipelines
from sklearn.pipeline import Pipeline #Making Pipelines
from sklearn.compose import ColumnTransformer

import sys, os
from dataclasses import dataclass
import pandas as pd, numpy as np

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object



#Data Transformation Config
@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")



#Data Transformation class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation intitiated")
            categorical_cols = ["cut", "color", "clarity"]
            numerical_cols = ["carat", "depth", "table", "x", "y", "z"]

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline inititated.")

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                    
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info("Pipeline built")

            return preprocessor

            

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data")
            logging.info("Train DataFrame Head: "+ str(train_df.head()))
            logging.info("Test Dataframe Head: "+ str(test_df.head()))

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = "price"
            drop_columns = [target_column_name, "id"]

            logging.info("Spliting the Train and Test Dataframe into Input Train and Input Test Dataframe")

            input_feature_train_df = train_df.drop(columns= drop_columns, axis=1)
            target_feature_train = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test = test_df[target_column_name]
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing to training and test dataset")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            logging.info("Processor pickle is created and saved")
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            return(train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

            
        except Exception as e:
            logging.info(str(e))

            raise  CustomException(e,sys)
