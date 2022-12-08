from iqoptionapi.stable_api import IQ_Option
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import time
import matplotlib.pyplot as plt
from pickle import dump, load


# Função utilizada para criar a variável target, que indica se o preço da ação subiu ou desceu
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


# Seta a gpu, se tiver o tensorflow-gpu + CUDA configurados
def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        except RuntimeError as e:
            print(e)


# Classe que faz as operações na API, tanto GET quanto POST
class ApiData:

    def __init__(self, email, password, checkConnection=True, type='PRACTICE', otc=False):
        self.email = email
        self.password = password
        self.api = IQ_Option(self.email, self.password)
        self.api.connect()
        self.api.change_balance(type)  # PRACTICE / REAL
        self.actives = ['EURUSD', 'GBPUSD', 'EURJPY', 'EURGBP']
        self.actives_otc = [f'{moeda}-OTC' for moeda in self.actives]
        self.checkConnection = checkConnection
        self.otc = otc
        self.main = None
        self.profit = self.getProfit()['turbo']

    # Checa o status da conexão
    def connectionCheck(self):
        if not self.api.check_connect() and self.checkConnection:
            print('Conexão Inválida. Tentando novamente...')
            print(self.api)
            self.api.connect()
            self.api.change_balance(type)

    # Checa o saldo atuual
    def getBalance(self):
        return self.api.get_balance()

    # Checa se o ativo está aberto
    def is_open(self):
        open = self.api.get_all_open_time()
        if self.otc:
            open = open['turbo'][self.actives_otc[0]]['open']
        else:
            open = open['turbo'][self.actives[0]]['open']
        return open

    # Checa o resultado da última operação
    def getResult(self):
        message = self.api.get_optioninfo_v2(1)['msg']['closed_options'][0]['win']
        bet_size = self.api.get_optioninfo_v2(1)['msg']['closed_options'][0]['amount']
        win_amount = self.api.get_optioninfo_v2(1)['msg']['closed_options'][0]['win_amount']
        return message, bet_size, win_amount

    # Checa se os ativos estão abertos
    def getActives(self):
        return self.api.get_all_open_time()

    # Realiza a compra
    def buy(self, active, direction, amount):
        self.connectionCheck()
        complete, id = self.api.buy(amount, active, direction, 1)
        if complete:
            print('Compra realizada com sucesso!')
            print('ID da compra: ', id)
        else:
            print('Erro ao realizar compra!')
            print('ID da compra: ', id)

    # Recebe os dados históricos do ativo (Configurável para um período)
    def getHistoricalDataFrame(self, active, duration, limit):
        self.connectionCheck()
        return pd.DataFrame(self.api.get_candles(active, duration, limit, time.time()))

    # Implementação da função anterior, para pegar os dados de todos os ativos analisados e retornar em um único df
    def getData(self):
        self.connectionCheck()
        actives = self.actives_otc if self.otc else self.actives
        main = pd.DataFrame()
        for active in actives:
            print('Getting data for', active)
            if active == actives[0]:
                main = self.getHistoricalDataFrame(active, 60, 1000).drop(columns={'from', 'to', 'id', 'at'})
            else:
                current = self.getHistoricalDataFrame(active, 60, 1000).drop(
                    columns={'from', 'to', 'open', 'min', 'max', 'id', 'at'})
                current.columns = [f'close_{active}', f'volume_{active}']
                main = main.join(current)
        return main

    # Retorna o valor que o ativo desejado está atualmente pagando ao ganhar
    def getProfit(self):
        active = self.actives[0] if not self.otc else self.actives_otc[0]
        return self.api.get_all_profit()[active]


# Aqui acontece o feature engineering, onde são criadas as variáveis que serão utilizadas para treinar o modelo
class ModelData:

    def __init__(self, apidata, sequence_length=15, future_period=2):
        self.ApiData = apidata
        self.sequence_length = sequence_length
        self.future_period = future_period
        self.scaler = MinMaxScaler()
        set_gpu()

    # Função que cria a variável target e outros indicadores financeiros usados na análise técnica
    def feature_engineering(self):
        self.ApiData.connectionCheck()
        self.data = self.ApiData.getData()
        df = self.data.copy()
        df['future'] = df['close'].shift(-self.future_period)
        df['target'] = list(map(classify, df['close'], df['future']))
        df['MA_20'] = df['close'].rolling(20).mean()
        df['MA_50'] = df['close'].rolling(50).mean()
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

    # Similar à função anterior, porém sem a variável target, para ser usada na predição
    def feature_engineering_predict(self):
        self.ApiData.connectionCheck()
        df = self.data.copy()
        df['MA_20'] = df['close'].rolling(20).mean()
        df['MA_50'] = df['close'].rolling(50).mean()
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

    # Função que cria as sequências para a RNN e separa os dados em treino e teste
    def preprocess_train(self):
        df = self.feature_engineering()
        times = sorted(df.index.values)
        last_5pct = times[-int(0.05 * len(times))]
        validation_df = df[(df.index >= last_5pct)]
        df = df[(df.index < last_5pct)]
        train_x, train_y = [], []
        for i in range(self.sequence_length, len(df) + 1):
            train_x.append(np.array(df.iloc[i - self.sequence_length:i].drop(['target'], axis=1)))
            train_y.append(df.iloc[i - 1]['target'])
        validation_x, validation_y = [], []
        for i in range(self.sequence_length, len(validation_df) + 1):
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


# Classe onde a rede neural é criada e treinada
class NeuralNetwork(ModelData):

    def __init__(self, apidata, sequence_length=15, future_period=2, otc=False):
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

    ''' 
        Criação e fit do modelo LSTM, que é salvo em um arquivo .h5. Foi utilizada a função relu como função de ativação,
    a fim de se evitar o problema do vanishing gradient. A função de ativação softmax foi utilizada na última camada,
    pois o problema é de classificação binária. A função de perda ideal é a binary crossentropy, também devido à
    característica binária do problema, mas aparentemente a sparse_categorical_crossentropy ofereceu melhores resultados. 
    A métrica utilizada foi a acurácia, pois é a principal varável que resolve o problema original. Em geral, a performance 
    do modelo está péssima. No treino/validação ele oferece acurácia de até 56%, porém em produção a acurácia fica em torno
    de 50% (que é o esperado em um problema de classificação binária). O mesmo ocorre com o modelo em que mais dados são
    utilizados. Estou implementando o AutoKeras para testar, porém os resultados continuam similares. 
    '''

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

    # Carrega o checkpoint do modelo. Caso não exista um, cria um novo.
    def load_model_checkpoint(self):
        try:
            model = tf.keras.models.load_model(self.model_path)
            print('Model loaded')
        except:
            model = self.lstm_model()
            print('Model created')
        return model

    # Retreina o modelo
    def retrain(self):
        self.model = self.lstm_model()

    # Preprocessamento dos dados a serem preditos. A função está aqui para que o scaler seja carregado corretamente.
    def preprocess_predict_data(self):
        df = self.feature_engineering_predict()
        test_x = []
        for i in range(self.sequence_length, len(df) + 1):
            test_x.append(np.array(df.iloc[i - self.sequence_length:i]))
        test_x = np.array(test_x)
        test_x = self.scaler.transform(test_x.reshape(-1, test_x.shape[2])).reshape(test_x.shape)
        test_x = test_x[len(test_x) - 1:len(test_x)]
        print('Predict data shape:', test_x.shape)
        return test_x

    # Realiza a predição
    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction

    # Avalia o modelo
    def evaluate(self, data, labels):
        score = self.model.evaluate(data, labels)
        return score

    # Salva o scaler
    def save_model_scaler(self, model, scaler):
        model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            dump(scaler, f)

    # Carrega o scaler
    def load_model_scaler(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = load(open(self.scaler_path, 'rb'))
            print('Model loaded')
        except:
            self.model = self.lstm_model()
            print('Model created')

    # Plota o gráfico de acurácia
    def plot_val_accuracy(self):
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.show()

    # Plota o gráfico de perda
    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
