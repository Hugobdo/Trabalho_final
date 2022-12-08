from api_data import classify, createFeatures
from mongo_connector import MongoData
from sklearn.model_selection import train_test_split
import autokeras as ak
import tensorflow as tf


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        except RuntimeError as e:
            print(e)


class AutoKeras:

    def __init__(self, active='EURUSD', max_trials=15, overwrite=False):
        self.mongo = MongoData(active)
        self.data = self.mongo.read_all_data()
        self.max_trials = max_trials
        self.overwrite = overwrite
        self.processed_data = None
        self.automodel_classifier = None
        self.automodel_regression = None

    def feature_engineering_classifier(self):
        self.processed_data = createFeatures(self.data)
        self.processed_data['target'] = self.processed_data['close'].shift(-1)
        self.processed_data['target'] = list(map(classify, self.processed_data['close'], self.processed_data['target']))
        self.processed_data = self.processed_data.dropna()
        drop_columns = ['from', 'from_datetime', 'at', 'to', 'id', '_id']
        self.processed_data = self.processed_data.drop(drop_columns, axis=1)
        x = self.processed_data.drop('target', axis=1)
        y = self.processed_data['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def feature_engineering_regressor(self):
        self.processed_data = createFeatures(self.data)
        self.processed_data['target'] = self.processed_data['close'].shift(-1)
        self.processed_data = self.processed_data.dropna()
        drop_columns = ['from', 'from_datetime', 'at', 'to', 'id', '_id']
        self.processed_data = self.processed_data.drop(drop_columns, axis=1)
        x = self.processed_data.drop('target', axis=1)
        y = self.processed_data['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def fit(self, rnn=False):
        if rnn:
            self.automodel_regression = ak.StructuredDataRegressor(
                max_trials=self.max_trials,
                overwrite=self.overwrite,
                loss='mean_squared_error',
                metrics=['mean_squared_error']
            )
            x_train, x_test, y_train, y_test = self.feature_engineering_regressor()
            self.automodel_regression.fit(x_train, y_train)

        else:
            self.automodel_classifier = ak.StructuredDataClassifier(
                max_trials=self.max_trials,
                overwrite=self.overwrite,
                loss='binary_crossentropy'
            )
            x_train, x_test, y_train, y_test = self.feature_engineering_classifier()
            self.automodel_classifier.fit(x_train, y_train)


search = AutoKeras('EURUSD', 15, overwrite=True)
search.fit(rnn=True)
