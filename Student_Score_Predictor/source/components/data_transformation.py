import os, sys
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from logger import logger
from exception import CustomException
import numpy as np

from source.components.data_ingestion import DataIngestion
from source.utils.common import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Function used for data transformation 
    def get_data_transformer(self):

        try:
            numerical_columns = ["reading_score", "writing_score"]

            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            logger.info(f"Numerical Columns: {numerical_columns}")
            logger.info(f"Categorical Columns: {categorical_columns}")

            numerical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("Encoding", OneHotEncoder(handle_unknown='ignore')),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )

            transformer = ColumnTransformer(
                [
                    ("Numerical pipeline", numerical_pipeline, numerical_columns),
                    ("Categorical pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            logger.info("Numerical data preprocessing has completed.")

            logger.info("Categorical columns encoding and processing has completed.")

            return transformer
        
        except Exception as e:
            logger.info(e)
            raise CustomException(e, sys)
        
    def data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            logger.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logger.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logger.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_file,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file,
            )
        
        except Exception as e:
            logger.info(e)
            raise CustomException(e,sys)
        


        
    