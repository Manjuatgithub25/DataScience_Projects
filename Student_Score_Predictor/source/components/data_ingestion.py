from dataclasses import dataclass
import os, sys
from logger import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifact", "data.csv")
    train_data_path = os.path.join("artifact", "train.csv")
    test_data_path = os.path.join("artifact", "test.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

        logger.info("Data ingestion has started")

    def data_ingestion(self):
        try:
            df = pd.read_csv("StudentPerformance.csv")
            logger.info("Read the data as dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logger.info("Data split has been performed and files are stored")

            return (
                self.data_ingestion_config.raw_data_path,
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)



        
