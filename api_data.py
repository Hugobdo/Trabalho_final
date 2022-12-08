from iqoptionapi.stable_api import IQ_Option
import pandas as pd
import time
from datetime import datetime


def unix_to_date(unix):
    return datetime.fromtimestamp(unix)


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def createFeatures(df):
    df['Hour'] = df['from_datetime'].dt.hour
    df['Minute'] = df['from_datetime'].dt.minute
    df['Second'] = df['from_datetime'].dt.second
    df['Day'] = df['from_datetime'].dt.day
    df['Month'] = df['from_datetime'].dt.month
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
    df.dropna(inplace=True)

    return df


class ApiData:

    def __init__(self, email, password, checkConnection=True, type='PRACTICE', otc=False, data_rows=1000):
        self.email = email
        self.password = password
        self.api = IQ_Option(self.email, self.password)
        self.api.connect()
        self.api.change_balance(type)  # PRACTICE / REAL
        self.actives = ['EURUSD', 'GBPUSD', 'EURJPY', 'EURGBP']
        self.actives_otc = [f'{moeda}-OTC' for moeda in self.actives]
        self.active = self.actives[0] if not otc else self.actives_otc[0]
        self.checkConnection = checkConnection
        self.otc = otc
        self.profit = self.getProfit()['turbo']
        self.data_rows = data_rows

    def connectionCheck(self):
        if not self.api.check_connect() and self.checkConnection:
            print('Conexão Inválida. Tentando novamente...')
            print(self.api)
            self.api.connect()
            self.api.change_balance(type)

    def getBalance(self):
        return self.api.get_balance()

    def getResult(self):
        message = self.api.get_optioninfo_v2(1)['msg']['closed_options'][0]['win']
        bet_size = self.api.get_optioninfo_v2(1)['msg']['closed_options'][0]['amount']
        win_amount = self.api.get_optioninfo_v2(1)['msg']['closed_options'][0]['win_amount']
        return message, bet_size, win_amount

    def getActives(self):
        return self.api.get_all_open_time()

    def buy(self, active, direction, amount):
        self.connectionCheck()
        complete, id = self.api.buy(amount, active, direction, 1)
        if complete:
            print('Compra realizada com sucesso!')
            print('ID da compra: ', id)
        else:
            print('Erro ao realizar compra!')
            print('ID da compra: ', id)

    def getHistoricalDataFrame(self, active, duration, limit):
        self.connectionCheck()
        return pd.DataFrame(self.api.get_candles(active, duration, limit, time.time()))

    def getData(self, size=1000):
        self.connectionCheck()
        actives = self.actives_otc if self.otc else self.actives
        main = pd.DataFrame()
        for active in actives:
            print('Getting data for', active)
            if active == actives[0]:
                main = self.getHistoricalDataFrame(active, 60, size).drop(columns={'from', 'to', 'id', 'at'})
            else:
                current = self.getHistoricalDataFrame(active, 60, size).drop(
                    columns={'from', 'to', 'open', 'min', 'max', 'id', 'at'})
                current.columns = [f'close_{active}', f'volume_{active}']
                main = main.join(current)
        return main

    def getProfit(self):
        active = self.actives[0] if not self.otc else self.actives_otc[0]
        return self.api.get_all_profit()[active]
