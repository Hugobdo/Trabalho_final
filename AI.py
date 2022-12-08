import numpy as np
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
from pickle import dump, load
from api_data import classify, set_gpu



class ModelData:

    def __init__(self, apidata, sequence_length=15, future_period=1):
        self.data = None
        self.ApiData = apidata
        self.sequence_length = sequence_length
        self.future_period = future_period
        self.scaler = MinMaxScaler()
        set_gpu()

    def feature_engineering(self):
        self.ApiData.connectionCheck()
        self.data = self.ApiData.getData()
        df = self.data.copy()
        df['future'] = df['close'].shift(-self.future_period)
        df['target'] = list(map(classify, df['close'], df['future']))
        df['MA_9'] = df['close'].rolling(9).mean()
        df['MA_21'] = df['close'].rolling(21).mean()
        df['MA_72'] = df['close'].rolling(72).mean()
        df['Stdev_20'] = df['close'].rolling(20).std()
        df['Stdev_50'] = df['close'].rolling(50).std()
        df['L14'] = df['close'].rolling(14).min()
        df['H14'] = df['close'].rolling(14).max()
        df['K'] = 100 * (df['close'] - df['L14']) / (df['H14'] - df['L14'])
        df['D'] = df['K'].rolling(3).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        df['EMA_50'] = df['close'].ewm(span=50).mean()
        df['RSI'] = 100 - 100 / (1 + df['close'].diff(1).rolling(14).mean() / df['close'].diff(1).rolling(14).std())
        df['ROC'] = df['close'].diff(1) / df['close'].shift(1)
        df['Momentum'] = df['close'] - df['close'].shift(1)
        df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        df['OBV'] = df['close'].diff(1).apply(lambda x: x if x > 0 else 0).cumsum()
        df['gain'] = df['close'].diff(1).apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['close'].diff(1).apply(lambda x: x if x < 0 else 0)
        df = df.drop(columns={'open', 'min', 'max', 'L14', 'H14', 'gain', 'loss', 'future'}).dropna()
        return df

    def feature_engineering_predict(self):
        self.ApiData.connectionCheck()
        df = self.data.copy()
        df['MA_9'] = df['close'].rolling(9).mean()
        df['MA_21'] = df['close'].rolling(21).mean()
        df['MA_72'] = df['close'].rolling(72).mean()
        df['Stdev_20'] = df['close'].rolling(20).std()
        df['Stdev_50'] = df['close'].rolling(50).std()
        df['L14'] = df['close'].rolling(14).min()
        df['H14'] = df['close'].rolling(14).max()
        df['K'] = 100 * (df['close'] - df['L14']) / (df['H14'] - df['L14'])
        df['D'] = df['K'].rolling(3).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        df['EMA_50'] = df['close'].ewm(span=50).mean()
        df['RSI'] = 100 - 100 / (1 + df['close'].diff(1).rolling(14).mean() / df['close'].diff(1).rolling(14).std())
        df['ROC'] = df['close'].diff(1) / df['close'].shift(1)
        df['Momentum'] = df['close'] - df['close'].shift(1)
        df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        df['OBV'] = df['close'].diff(1).apply(lambda x: x if x > 0 else 0).cumsum()
        df['gain'] = df['close'].diff(1).apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['close'].diff(1).apply(lambda x: x if x < 0 else 0)
        df = df.drop(columns={'open', 'min', 'max', 'L14', 'H14', 'gain', 'loss'}).dropna()
        return df

    def preprocess_train(self):
        df = self.feature_engineering()
        times = sorted(df.index.values)
        last_5pct = times[-int(0.05 * len(times))]
        validation_df = df[(df.index >= last_5pct)]
        df = df[(df.index < last_5pct)]
        train_x, train_y = [], []
        for i in range(self.sequence_length, len(df)+1):
            train_x.append(np.array(df.iloc[i - self.sequence_length:i].drop(['target'], axis=1)))
            train_y.append(df.iloc[i - 1]['target'])
        validation_x, validation_y = [], []
        for i in range(self.sequence_length, len(validation_df)+1):
            validation_x.append(np.array(validation_df.iloc[i - self.sequence_length:i].drop(['target'], axis=1)))
            validation_y.append(validation_df.iloc[i - 1]['target'])
        train_x, train_y = np.array(train_x), np.array(train_y)
        validation_x, validation_y = np.array(validation_x), np.array(validation_y)
        train_x = self.scaler.fit_transform(train_x.reshape(-1, train_x.shape[2])).reshape(train_x.shape)
        validation_x = self.scaler.transform(validation_x.reshape(-1, validation_x.shape[2])).reshape(
            validation_x.shape)
        print('Train data shape:', train_x.shape)
        print('Train labels shape:', train_y.shape)
        print('Validation data shape:', validation_x.shape)
        print('Validation labels shape:', validation_y.shape)
        return train_x, train_y, validation_x, validation_y


class NeuralNetwork(ModelData):

    def __init__(self, apidata, sequence_length=15, future_period=1, otc=False):
        super().__init__(apidata, sequence_length, future_period)
        self.otc = otc
        self.name = f'{self.sequence_length}-SEQ-{self.future_period}-PRED-{int(time.time())}'  # Model Name for logging
        self.tensorboard = TensorBoard(log_dir=f'logs/{self.name}')  # Tensorboard
        self.filepath = "LSTM-best"  # Model Checkpoint for saving best model
        self.otc_filepath = "LSTM-otc-best"  # Model Checkpoint for saving best otc model
        self.model_path = f'models/{self.filepath}.h5' if not self.otc else f'models/{self.otc_filepath}.h5'
        self.scaler_path = f'models/{self.filepath}_scaler.pkl' if not self.otc else f'models/{self.otc_filepath}_scaler.pkl'
        self.checkpoint_path = f'models/{self.filepath}.h5' if not self.otc else f'models/{self.otc_filepath}.h5'
        self.checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          mode='max')  # Model Checkpoint
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')  # Early Stopping
        self.optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)
        self.history = None
        self.model = None
        self.scaler = None
        self.load_model_scaler()

    def lstm_model(self):
        print('Building LSTM model...')
        train_x, train_y, validation_x, validation_y = self.preprocess_train()
        model = Sequential()
        model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        self.history = model.fit(train_x, train_y, batch_size=64, epochs=100,
                                 validation_data=(validation_x, validation_y),
                                 callbacks=[self.tensorboard, self.checkpoint, self.early_stopping])
        self.save_model_scaler(model, self.scaler)
        print('LSTM model built successfully.')
        return model

    def load_model_checkpoint(self):
        try:
            model = tf.keras.models.load_model(self.model_path)
            print('Model loaded')
        except:
            model = self.lstm_model()
            print('Model created')
        return model

    def retrain(self):
        self.model = self.lstm_model()

    def preprocess_predict_data(self):
        df = self.feature_engineering_predict()
        test_x = []
        for i in range(self.sequence_length, len(df)+1):
            test_x.append(np.array(df.iloc[i - self.sequence_length:i]))
        test_x = np.array(test_x)
        test_x = self.scaler.transform(test_x.reshape(-1, test_x.shape[2])).reshape(test_x.shape)
        test_x = test_x[len(test_x) - 1:len(test_x)]
        print('Predict data shape:', test_x.shape)
        return test_x

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction

    def evaluate(self, data, labels):
        score = self.model.evaluate(data, labels)
        return score

    def save_model_scaler(self, model, scaler):
        model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            dump(scaler, f)

    def load_model_scaler(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = load(open(self.scaler_path, 'rb'))
            print('Model loaded')
        except:
            self.model = self.lstm_model()
            print('Model created')

    def plot_val_accuracy(self):
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.show()

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
