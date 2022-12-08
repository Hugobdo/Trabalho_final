from api_data import ApiData, createFeatures
from datetime import datetime
import time
from tqdm import tqdm
import autokeras as ak
import tensorflow as tf
from mongo_connector import MongoData


class Main:
    def __init__(self,
                 user, password,
                 bet_money_base, martingale_base,
                 martingale_growth, stop_loss,
                 min_profit=0.82,
                 sequence_length=15,
                 type='PRACTICE',
                 otc=False):
        self.min_profit = min_profit
        self.user = user
        self.password = password
        self.otc = otc
        self.type = type
        self.bet_money_base = bet_money_base
        self.martingale_base = martingale_base
        self.martingale_growth = martingale_growth
        self.stop_loss = stop_loss
        self.sequence_length = sequence_length
        self.martingale = 1
        self.lose_streak = 0
        self.bet_money = self.bet_money_base
        self.log_name = 'log.txt'
        self.runtime_log = 'runtime_log.txt'
        self.log = '\nStarting program...'
        self.api = ApiData(self.user, self.password, type=self.type, otc=self.otc)
        self.classifier_model = tf.keras.models.load_model(r'structured_data_classifier\best_model',
                                                           custom_objects=ak.CUSTOM_OBJECTS)
        self.regressor_model = tf.keras.models.load_model(r'structured_data_regressor\best_model',
                                                          custom_objects=ak.CUSTOM_OBJECTS)
        self.active = self.api.actives[0] if not self.otc else self.api.actives_otc[0]
        self.mongo = MongoData(ApiData=self.api, otc=self.otc)
        self.data = None
        self.bet = True
        self.runtime_logger()

    def runtime_logger(self):
        now = datetime.now()
        with open(self.runtime_log, 'a') as f:
            f.write(f'{self.log}_{now}\n')

    def predict_classifier(self):
        self.log = 'Classifier predicting...'
        print(self.log)
        self.runtime_logger()

        self.mongo.update()
        self.data = self.mongo.last_entries(72)
        self.data = createFeatures(self.data)
        drop_columns = ['from', 'from_datetime', 'at', 'to', 'id', '_id']
        self.data = self.data.drop(drop_columns, axis=1)
        prediction = self.classifier_model.predict(self.data)
        put_prob = prediction[0][0]
        call_prob = 1 - prediction[0][0]
        return call_prob, put_prob

    def predict_regressor(self):
        self.log = 'Regressor predicting...'
        print(self.log)
        self.runtime_logger()

        self.mongo.update()
        self.data = self.mongo.last_entries(72)
        self.data = createFeatures(self.data)
        drop_columns = ['from', 'from_datetime', 'at', 'to', 'id', '_id']
        self.data = self.data.drop(drop_columns, axis=1)
        prediction = self.regressor_model.predict(self.data)
        return prediction

    def trade(self, call_prob, put_prob):
        self.log = 'Trading...'
        print(self.log)
        self.runtime_logger()

        if call_prob > put_prob:
            self.log = 'Call'
            print(self.log)
            self.runtime_logger()
            self.api.buy(self.active, 'Call', self.bet_money)
            return 'Call'
        elif call_prob < put_prob:
            self.log = 'Put'
            print(self.log)
            self.runtime_logger()
            self.api.buy(self.active, 'Put', self.bet_money)
            return 'Put'
        else:
            self.log = 'Same chance. Will pass...'
            print(self.log)
            self.runtime_logger()
            return 'Pass'

    def check_result(self):
        self.log = 'Retrieving results...'
        print(self.log)
        self.runtime_logger()

        message, bet_size, win_amount = self.api.getResult()
        message = 'equal' if bet_size == win_amount else message
        return message, bet_size, win_amount

    def evaluate_results(self, message):
        self.log = 'Evaluating results...'
        print(self.log)
        self.runtime_logger()

        if message == 'win':
            self.log = 'Win'
            print(self.log)
            self.runtime_logger()

            self.bet_money = self.bet_money_base
            self.lose_streak = 0

        elif message == 'loose':
            self.log = 'Loss'
            print(self.log)
            self.runtime_logger()

            self.martingale = self.martingale_base * (self.martingale_growth ** self.lose_streak)
            self.bet_money = self.bet_money * self.martingale
            self.lose_streak += 1

            self.log = f'Loss streak: {self.lose_streak}. Increasing bet...'
            print(self.log)
            self.runtime_logger()

            # Stop loss
            if self.lose_streak > self.stop_loss:
                self.log = 'Stop loss reached. Resetting streak...'
                print(self.log)
                self.runtime_logger()

                self.lose_streak = 0
                self.bet_money = self.bet_money_base

        elif message == 'equal':
            self.log = 'Equal'
            print(self.log)
            self.runtime_logger()

        else:
            self.log = 'Error'
            print(self.log)
            self.runtime_logger()

            exit(0)

    def balance(self):
        balance = self.api.getBalance()
        self.log = 'Checking balance...'
        print(self.log)
        self.runtime_logger()

        self.log = f'Balance: {balance}'
        print(self.log)
        self.runtime_logger()
        return balance

    def profitable(self):  # Checa se o retorno da corretora é maior ou igual ao mínimo esperado
        profit = self.api.getProfit()['turbo']
        profitable = True
        if profit < self.min_profit:
            self.log = 'Profit too low. Skipping...'
            profitable = False
            print(self.log)
            self.runtime_logger()
            time.sleep(1800)
        return profitable

    def timed_bet(self, wait=True):  # Faz uma aposta com base na probabilidade de cada resultado e espera o resultado.
        call_prob, put_prob = self.predict_classifier()
        print('Probability of PUT: ', put_prob)
        print('Probability of CALL: ', call_prob)

        if wait:
            print(f'Will wait {60 - datetime.now().second} seconds before placing bet')
            for _ in (pbar := tqdm(range(60 - datetime.now().second))):
                time.sleep(1)
                pbar.set_description(f'{60 - datetime.now().second} seconds left')

        op_type = self.trade(call_prob, put_prob)
        if op_type == 'Pass':
            print('Equal probability. Will pass...')
            probability = 0.5
            self.bet = False
        else:
            print('Bet done! Please wait for result...')
            for _ in (pbar := tqdm(range(60 - datetime.now().second))):  # PROGRESS BAR
                time.sleep(1)
                pbar.set_description(f'{60 - datetime.now().second} seconds left')
            self.bet = True
            probability = call_prob if op_type == 'Call' else put_prob
            self.log = f'Probability of {op_type}: {probability}'
            self.runtime_logger()
        return op_type, probability

    def bet_response(self, op_type, probability, wait=True):  # Checa o resultado da aposta e avalia o resultado.
        if wait:
            time.sleep(3)
        message, bet_size, win_amount = self.check_result()
        self.bet = True
        if message == '':
            self.bet = False
            return True
        balance = self.balance()
        profit = self.api.getProfit()['turbo']
        self.log = {
            'datetime': datetime.now(),
            'result': message,
            'bet': self.bet_money,
            'lose_streak': self.lose_streak,
            'martingale_base': self.martingale_base,
            'martingale_growth': self.martingale_growth,
            'martingale': self.martingale,
            'operation_type': op_type,
            'probability': probability,
            'strategy': 'Stop Loss',
            'active': self.active,
            'current_balance': balance,
            'operation_profit': profit,
            'min_profit': self.min_profit
        }
        self.mongo.logger(self.log)
        self.evaluate_results(message)
        return False

    # Aposta em todas as candles e avalia o resultado.
    def multi_bet(self):
        self.log = 'Starting...'
        print(self.log)
        self.runtime_logger()
        self.mongo.update()
        while True:
            otc_test = False
            if self.profitable():
                if datetime.now().second < 30:
                    op_type, probability = self.timed_bet()  # Primeira aposta. Garante começo do minuto - Tempo 1
                    if self.bet:
                        self.bet_response(op_type, probability)  # Checa o resultado da primeira aposta

                        while self.bet:
                            if self.profitable():
                                op_type, probability = self.timed_bet(wait=False)  # Faz a segunda aposta - Tempo 2
                                if self.bet:
                                    otc_test = self.bet_response(op_type, probability)  # Checa o resultado

                                    # Testa se o mercado operante ainda está aberto. Se não tiver, inverte OTC.
                                    if otc_test:
                                        self.log = 'Market closed. Changing OTC...'
                                        print(self.log)
                                        self.runtime_logger()
                                        break
            # Invertendo OTC
            if otc_test:
                self.__init__(
                    self.user,
                    self.password,
                    self.bet_money_base, self.martingale_base,
                    self.martingale_growth, self.stop_loss,
                    otc=not self.otc, sequence_length=self.sequence_length
                )


robot = Main(
    'ml.puc.teste@hotmail.com',
    'puc.1234',
    2,  # Bet money
    1.94430957791836,  # Martingale base
    1.32378405828788,  # Martingale growth
    3,  # Stop loss
    otc=False,
    sequence_length=5
)
robot.multi_bet()
