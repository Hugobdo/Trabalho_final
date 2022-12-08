from api_data import ApiData, unix_to_date
from tqdm import tqdm
import pymongo
import time
import pandas as pd
import json
import numpy


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(CustomEncoder, self).default(obj)


class MongoData:
    def __init__(self, active='EURUSD', ApiData=None, otc=False):
        self.client = pymongo.MongoClient('mongodb://localhost')
        self.db = self.client.binary_bot
        self.active = active
        self.active_collection = self.db[self.active] if not otc else self.db[f'{self.active}_OTC']
        self.api = ApiData
        self.otc = otc
        self.log_collection = self.db['operations']
        self.apidata = ApiData
        self.first_date = None
        self.last_date = None

    def insert_historical(self, rows):
        if self.apidata is None:
            raise ValueError('An Api connection is required to update the database!')
        self.first_date = list(self.active_collection.find().sort("from", pymongo.ASCENDING).limit(1))[0]["from"]
        end_date = self.first_date
        for i in tqdm(range(int(rows / 1000))):
            data = self.apidata.api.get_candles(self.active, 60, 1000, end_date)
            for candle in data:
                candle['from_datetime'] = unix_to_date(candle['from'])
            self.active_collection.insert_many(data)
            end_date = int(data[0]["from"]) - 1
        print('Done!')

    def update(self):
        if self.apidata is None:
            raise ValueError('An Api connection is required to update the database!')
        end_date = time.time()
        self.last_date = list(self.active_collection.find().sort("from", pymongo.DESCENDING).limit(1))[0]["from"]
        while end_date > self.last_date:
            data = self.apidata.api.get_candles(self.active, 60, 1, end_date)
            if data[-1]["from"] == self.last_date:
                break
            for candle in data:
                candle["from_datetime"] = unix_to_date(candle["from"])
            self.active_collection.insert_many(data)
            end_date = int(data[0]["from"]) - 1
            print(f'Inserted {data[0]["from_datetime"]} on database')
        print('Done!')

    def read_all_data(self):
        return pd.DataFrame.from_records(self.active_collection.find())

    def last_entries(self, n=100):
        return pd.DataFrame.from_records(self.active_collection.find().sort("from", pymongo.DESCENDING).limit(n))

    def logger(self, log):
        # noinspection PyTypeChecker
        store_datetime = log['datetime']
        data_dict = json.dumps(log, cls=CustomEncoder, default=str)
        data_dict_final = json.loads(data_dict)
        data_dict_final['datetime'] = store_datetime
        self.log_collection.insert_one(data_dict_final)


# mongo = MongoData(active='EURUSD', ApiData=ApiData('ml.puc.teste@hotmail.com', 'puc.1234', otc=False))
# mongo.insert_historical(1894000)