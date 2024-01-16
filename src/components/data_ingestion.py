import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import geopy
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 


from src.components.data_transformation import DataTransformation


# Initialize the Data ingestion configuration
@dataclass
class DataIngesionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

# Creating class for Data ingestion config
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngesionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df = pd.read_csv(os.path.join("notebooks/data/finalTrain.csv"))
            logging.info('Dataset read as pandas Dataframe')
            
            # Calculate distance and create a new column
            def calculate_distance(raw):
                restaurant_cords=(raw['Restaurant_latitude'],raw['Restaurant_longitude'])
                delivery_cords=(raw['Delivery_location_latitude'],raw['Delivery_location_longitude'])
                return geodesic(restaurant_cords,delivery_cords).kilometers

            df['distance (in km)']=df.apply(calculate_distance , axis=1)


            # Drop unnecessary columns
            df = df.drop(["ID", "Delivery_person_ID", "Delivery_person_Ratings", "Restaurant_latitude",
                "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude",
                "Order_Date", "Type_of_order","Time_Orderd","Time_Order_picked"], axis=1)
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('train test split')
            train_set, test_set = train_test_split(df, test_size=0.40, random_state=40)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.info('Exception occurred at Data Ingestion stage')
            raise CustomException(e, sys)
        

    # run data ingestion 
if __name__ == '__main__':
    obj=DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
 