#폴더생성 파일경로준비 등
import os
# 통화문자열 포맷
import locale
# 학습 과정 중 중에 정보를 기록하기 위함
import logging
#시간 값을 얻어오고 시간 문자열 포맷을 위해 사용
import time
import datetime
#자료 구조
import numpy as np
import pandas as pd
#투자 설정, 로깅 설정들을 하기 위한 모듈로 여러 상수 값 포함
import settings

from environment import Environment
from agent import Agent
from policy_network import  PolicyNetwork
from visualizer import Visualizer

logger = logging.getLogger(__name__)


class PolilcyLearner:
    # chart_data Environment객체 생시 넣어줌
    def __init__(self,stock_code,chart_data,training_data = None,min_trading_unit=1, max_trading_unit=2,delayed_reward_threshold=.05,lr=0.01):
        #종목코드
        self.stock_code = stock_code
        self.chart_data = chart_data
        #환경 객체
        self.environment = Environment(chart_data)
        #에이전트 객체
        self.agent = Agent(self.environment,min_trading_unit=min_trading_unit,max_trading_unit=max_trading_unit,delayed_reward_threshold=delayed_reward_threshold)
        #학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        #정책 신경망 : 입력크기 = 학습 데이터의 크기 + 에이전트 상태 크기
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.policy_network = PolicyNetwork(input_dim=self.num_features,output_dim=self.agent.NUM_ACTIONS, lr=lr)
        #가시화 모듈
        self.visualizer = Visualizer()

    #에포크마다 호출하는 reset함수
    def reset(self):
        self.sample = None
        # 학습데이터를 읽어가면서 1씩 증가하는 변수
        self.training_data_idx = -1

    #max_memory 배치 학습 데이터를 만들기 위해 과거 데이터를 저장할 배열 balance 에이전트 초기 투자 자본금
    def fit(self,num_epoches=1000, max_memory=60, balance=1000000, discount_factor=0, start_epsilon=.5, learning= True):
        logger.info("LR:{lr}, DF : {discount_factor}, TU : [{min_trading_unit}, {max_trading_unit}],"
                    "DRT: {delayed_reward_threshold}".format(lr=self.policy_network.lr,discount_factor=discount_factor,min_trading_unit,self.agent.min_trading_unit,max_trading_unit=self.agent.max_trading_unit,
                                                             delayed_reward_threshold=self.agent.delayed_reward_threshold))

        #가시화 준비
        #ckxm 차트 데이터는 변하지 않음으로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data)

        #가시화 결과 저장할 폴더 준비  폴더이름에 시간 날짜를 넣어 중복되는 일이 없도록 함
        epoch_summary_dir = os.path.join(settings.BASE_DIR,'epoch_summary/%s/epoch_summary_%s'%(self.stock_code,settings.timestr))
        if not os.path.isdir(epoch_summary_dir):
            os.makedirs(epoch_summary_dir)

        #에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        #학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            #에포크 관련 정보 초기화
            loss = 0
            #수행한 에포크 수
            itr_cnt = 0
            #수익이 발생한 에포크 수
            win_cnt = 0
            #무작위 투자를 수행한 횟수
            exploration_cnt = 0
            batch_size = 0
            #수익이 발생하여 긍정적 지연 보상을 준 수
            pos_learning_cnt = 0
            #손실이 발생하여 부정적 지연 보상을 준 수
            neg_learning_cnt = 0

            #메모리 초기화
            memory_sample = []
            memory_action = []
            memory_reward = []
            memory_prob = []
            memory_pv = []
            memory_num_stocks = []
            memory_exp_idx = []
            memory_learning_idx = []

            #환경, 에이전트, 정책 신경망 초기화
            self.environment.reset()
            self.agent.reset()
            self.policy_network.reset()
            self.reset()

            #가시화기 초기화 --> 2,3,4 번 차트 초기화 및 x축차트 범위 파라미터로 삽입
            self.visualizer.clear([0,len(self.chart_data)])

            #학습을 진행할수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon*(1. - float(epoch)/(num_epoches -1))
            else:
                epsilon = 0

            #하나의 에포크를 수행하는 while
            while True:
                #샘플 생성
                next_sample = self._build_sample()
                if next_sample is None:
                    break

                #정책 신경망 또는 탐험에 의한 행동 결정  return 결정한 행동,결정에 대한 확신도, 무작위 투자 유무
                action, confidence, exploration = self.agent.decide_action(self.policy_network,self.sample,epsilon)

                #결정한 행동을 수행하고 즉시 보상과 지현 보상 획득
                immediate_reward, delayed_reward = self.agent.act(action,confidence)

                #행동 및 행동에 대한 결과를 기억
                memory_sample.append(next_sample)
                memory_action.append(action)
                memory_reward.append(immediate_reward)
                memory_pv.append(self.agent.portfolio_value)
                memory_num_stocks.append(self.agent.num_stocks)
                #학습 데이터의 샘플,에이전트 행동,즉시보상,포트폴리오 가치, 보유 주식수를 저장--> 위에 추가한것들 모아2차원 배열 생성
                memory = [( memory_sample[i],memory_action[i],memory_reward[i])
                          for i in list(range(len(memory_action)))[-max_memory:]]
                if exploration:
                    #무작위 투자인 경우
                    memory_exp_idx.append(itr_cnt)
                    #정책 신경망의 출력을 그대로 저장하는 배열
                    memory_prob.append([np.nan]*Agent.NUM_ACTIONS)
                else:
                    #정책 신경망의 출력을 그대로 저장
                    memory_prob.append(self.policy_network.prob)

                #반복에 대한 정보 갱신
                batch_size += 1
                itr_cnt += 1
                #탐험을 한경우에만 증가
                exploration_cnt += 1 if exploration else  0
                # 지연 보상이 0보다 큰경우에만 1을 증가
                win_cnt += 1 if delayed_reward > 0 else 0

                #지연 보상이 발생한 경웨 학습을 수행하는 부분
                #학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                if delayed_reward == 0 and batch_size >= max_memory:
                    delayed_reward = immediate_reward
                if learning and delayed_reward != 0:
                    #배치 학습 데이터 크기  max_memory보다 작아야 함
                    batch_size = min(batch_size, max_memory)
                    #배치 학습 데이터 생성
                    x, y = self.__get_batch(memory,batch_size,discount_factor,delayed_reward)

                    if len(x) >0:
                        #긍부정 학습횟수 체크
                        if delayed_reward > 0:
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1
                        #정책 신경망 갱신
                        loss += self.policy_network.train_on_batch(x,y)
                        memory_learning_idx.append([itr_cnt,delayed_reward])
                    batch_size =0

            # 에포크 관련 정보 가시화
            #총에포크수 문자열 길이 체크 ex 1000번이면 4
            num_epoches_digit = len(str(num_epoches))
            #현제 에포크수를 num_epoches_digit 자릿수로 만들어줌 4이고 1epoch 이면 0001 이런식으로
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')

            self.visualizer.plot(
                #가시화기의 plot함수를 호출하여 에포크 수행 결과를 가시화
                epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                action_list=Agent.ACTIONS, actions=memory_action,
                num_stocks=memory_num_stocks, outvals=memory_prob,
                exps=memory_exp_idx, learning=memory_learning_idx,
                initial_balance=self.agent.initial_balance, pvs=memory_pv
            )
            #수행 결과를 파일로 저장
            self.visualizer.save(os.path.join(
                epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                    settings.timestr, epoch_str)))

            # 에포크 관련 정보 로그 기록
            if pos_learning_cnt + neg_learning_cnt > 0:
                loss /= pos_learning_cnt + neg_learning_cnt
            logger.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
                        "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                        "#Stocks:%d\tPV:%s\t"
                        "POS:%s\tNEG:%s\tLoss:%10.6f" % (
                            epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,
                            locale.currency(self.agent.portfolio_value, grouping=True),
                            pos_learning_cnt, neg_learning_cnt, loss))

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 최종 학습 결과 통꼐 정보 로그 기록 부분
        logger.info("Max PV: %s, \t # Win: %d" % (
            locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt))



    #미니 배치 데이터 생성 함수 부분
    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        #일련의 학습 데이터 및 에이전트 상태 배치 데이터 크기,학습 데이터 특징 크기 2차원
        x = np.zeros((batch_size, 1, self.num_features))
        # 일련의 지연보상 데이터 크기, 정책 신경ㅁ아의 결정하는 에이전트의 행동의 수  2차원
        # 배치 데이터 ㅡ키근ㄴ 지연 보상이 발생될 때 결정되기 때문에 17과 2로 고정
        y = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)

        for i, (sample, action, reward) in enumerate(
                reversed(memory[-batch_size:])):
            x[i] = np.array(sample).reshape((-1, 1, self.num_features))
            y[i, action] = (delayed_reward + 1) / 2
            if discount_factor > 0:
                y[i, action] *= discount_factor ** i
        return x, y

    #학습 데이터 샘플 생성 부분
    def _build_sample(self):
        # 차트 데이터의 현제 인덱스에서 다음 인덱스 데이터를 읽음
        self.environment.observe()
        #학습 데이터의 다음 인덱스가 존재하는지 확인
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    # 투자 시뮬레이션을 하는 trade 함수
    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)
        self.fit(balance=balance, num_epoches=1, learning=False)