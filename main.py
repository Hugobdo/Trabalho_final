from robot import ApiData, NeuralNetwork
from datetime import datetime
import time
from tqdm import tqdm


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
        self.bet_money = self.bet_money_base
        self.log_name = 'log.txt'
        self.runtime_log = 'runtime_log.txt'
        self.log = '\nStarting program...'
        self.api = ApiData(self.user, self.password, type=self.type, otc=self.otc)
        self.model = NeuralNetwork(self.api, sequence_length=self.sequence_length, future_period=2, otc=self.otc)
        self.active = self.api.actives[0] if not self.otc else self.api.actives_otc[0]
        self.lose_streak = 0
        self.bet = True
        self.runtime_logger()

    def operation_logger(self):
        now = datetime.now()
        with open(self.log_name, 'a') as f:
            f.write(f'{self.log}\n')

    def runtime_logger(self):
        now = datetime.now()
        with open(self.runtime_log, 'a') as f:
            f.write(f'{self.log}_{now}\n')

    def retrain(self):
        self.log = 'Retraining model...'
        print(self.log)
        self.runtime_logger()

        self.model.retrain()
        self.bet = False # Main loop. Reinicia o loop após retreinar

    def predict(self):
        self.log = 'Predicting...'
        print(self.log)
        self.runtime_logger()

        self.model.data = self.api.getData()  # Get new data. Must improve for faster fetching
        predict_data = self.model.preprocess_predict_data()
        prediction = self.model.predict(predict_data)
        call_prob = prediction[0][0]
        put_prob = prediction[0][1]
        return call_prob, put_prob

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
                self.log = 'Stop loss reached. Retraining model and resetting streak...'
                print(self.log)
                self.runtime_logger()

                self.retrain()
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

    def timed_bet(self, wait=True): # Faz uma aposta com base na probabilidade de cada resultado e espera o resultado.
        call_prob, put_prob = self.predict()
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

    def bet_response(self, op_type, probability, wait=True): # Checa o resultado da aposta e avalia o resultado.
        if wait:
            time.sleep(3)
        message, bet_size, win_amount = self.check_result()
        if message == '':
            return True
        balance = self.balance()
        profit = self.api.getProfit()['turbo']
        self.log = [
            datetime.now(),
            message,
            self.bet_money,
            self.lose_streak,
            self.martingale_base,
            self.martingale_growth,
            self.martingale,
            op_type,
            probability,
            'Stop Loss',
            self.active,
            balance,
            profit,
            self.min_profit
        ]
        self.operation_logger()
        self.evaluate_results(message)
        return False

    # Deprecated. Executava somente metade das candles
    def run(self):
        self.log = 'Starting...'
        print(self.log)
        self.runtime_logger()

        self.retrain()
        call_prob, put_prob, probability = 0.5, 0.5, 0.5
        i = 0

        while True:
            profit = self.api.getProfit()['turbo']
            if profit < self.min_profit:
                self.log = 'Profit too low. Skipping...'
                print(self.log)
                self.runtime_logger()
                time.sleep(1800)
                continue

            if datetime.now().second < 29 and i % 2 == 0:
                call_prob, put_prob = self.predict()
                print('Probability of PUT: ', put_prob)
                print('Probability of CALL: ', call_prob)
                print(f'Will wait {60 - datetime.now().second} seconds before placing bet')

                for _ in (pbar := tqdm(range(58 - datetime.now().second))):  # PROGRESS BAR
                    time.sleep(1)
                    pbar.set_description(f'{58 - datetime.now().second} seconds left')

                i += 1

            if datetime.now().second == 59 and i % 2 == 1:
                op_type = self.trade(call_prob, put_prob)

                if op_type == 'Pass':
                    print('Equal probability. Will pass...')
                    bet = False
                else:
                    print('Bet done! Please wait for result...')
                    for _ in (pbar := tqdm(range(60))):  # PROGRESS BAR
                        time.sleep(1)
                        pbar.set_description(f'{60 - datetime.now().second} seconds left')
                    bet = True
                    probability = call_prob if op_type == 'Call' else put_prob
                    self.log = f'Probability of {op_type}: {probability}'
                    self.runtime_logger()

                i += 1

                if bet:
                    tempo = datetime.now().second
                    while (tempo != 3):  # wait till 1 to see if win or lose
                        tempo = datetime.now().second

                    message, bet_size, win_amount = self.check_result()
                    balance = self.balance()
                    self.log = [
                        datetime.now(),
                        message,
                        self.bet_money,
                        self.lose_streak,
                        self.martingale_base,
                        self.martingale_growth,
                        self.martingale,
                        op_type,
                        probability,
                        'Stop Loss',
                        self.active,
                        balance,
                        profit,
                        self.min_profit
                    ]
                    self.operation_logger()
                    self.evaluate_results(message)

    # Aposta em todas as candles e avalia o resultado.
    def multi_bet(self):
        self.log = 'Starting...'
        print(self.log)
        self.runtime_logger()
        self.retrain()
        while True:
            otc_test = False
            if self.profitable():
                if datetime.now().second < 30:
                    op_type, probability = self.timed_bet() # Primeira aposta. Garante começo do minuto - Tempo 1
                    if self.bet:
                        self.bet_response(op_type, probability) # Checa o resultado da primeira aposta

                        while self.bet:
                            if self.profitable():
                                op_type, probability = self.timed_bet(wait=False) # Faz a segunda aposta - Tempo 2
                                if self.bet:
                                    otc_test = self.bet_response(op_type, probability) # Checa o resultado

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
                    self.bet_money_base,self.martingale_base,
                    self.martingale_growth,self.stop_loss,
                    otc = not self.otc, sequence_length=self.sequence_length
                )

robot = Main(
    'ml.puc.teste@hotmail.com',
    'puc.1234',
    2,
    2.1165,
    1.2023,
    3,
    otc=False,
    sequence_length=5
)
robot.multi_bet()
