import numpy as np
from keras.models import Sequential
from keras.layers import Activation,LSTM,Dense,BatchNormalization
from keras.optimizers import  sgd
from keras.optimizers import Adam
from keras.optimizers import Adadelta

class PolicyNetwork:

    def __init__(self,input_dim=0,output_dim=0,lr=0.01):
        self.input_dim = input_dim
        self.lr=lr

        #LSTM 신경망
        self.model = Sequential()

        self.model.add(LSTM(256,input_shape=(1,input_dim),return_sequences=True,stateful=False,dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256,return_sequences=True,stateful=False,dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256,return_sequences=False,stateful=False,dropout=0.5))
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))
        # print("help")
        # print(help(Adadelta))
        # adadelta = Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)
        # adam = Adam(lr=self.lr,beta_1=0.9,beta_2=0.999)
        # self.model.compile(optimizer=Adam(lr=self.lr,beta_1=0.9,beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),metrics=['accuracy'],loss='mse')
        self.model.compile(optimizer=sgd(lr=lr),loss='mse')
        self.prob = None

    #prob 초기화
    def reset(self):
        self.prob = None

    # 학습 데이터와 에이전트 상태를 합한 차원의 입력을 받아 매수,매도가 수익을 높일 확률을 구함, 여러 샘플을 받아 신경망의 출력을 반환
    def predict(self,sample):
        self.prob = self.model.predict(np.array(sample).reshape((1,-1,self.input_dim)))[0]
        return  self.prob

    #학습데이터 집합 x 레이블 y 로 학습
    def train_on_batch(self,x,y):
        return self.model.train_on_batch(x,y)

    #학습한 신경망을 파일로 저장
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path)

    #저장한 신경망을 불러움
    def load_model(self,model_path):
        if model_path is not None:
            self.model.load_weights(model_path)