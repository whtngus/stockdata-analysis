import numpy as np

class Agent:
    #에이전트 상태가 구성하는 값 개수
    #주식 보유 비율, 포트폴리오 가치 비율
    STATE_DIM = 2

    #매매 수수료 및 세금
    TRADING_CHARGE = 0 # 거래 수수료 미고려 0.015
    TRADING_TAX = 0 #거래세 미고려 0.3

    #행동
    ACTION_BUY = 0 #매수
    ACTION_SELL = 1 #매도
    ACTION_HOLD = 2 #관망
    ACTIONS = [ACTION_BUY,ACTION_SELL] # 인공 신경망에서 확률을 구할 행동들
    MUM_ACTIONS = len(ACTIONS) # 인공 신경망에서 고려할 출력값의 개수

    # environment객체 최소 최대 매매단위
    def __init__(self,environment,min_trading_unit=1,max_trading_unit =2 ,delayed_reward_threshold = .05):
        #Environment 객체
        #현제 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        #최소 매매 단위, 최대 매매 단위, 지연 보상 임계치
        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit
        self.delayed_reward_threshold = delayed_reward_threshold

        #Agent 클래스의 속성
        self.initial_balance = 0 #초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0 # 보유 주식수
        self.portfolio_value = 0 # balance + num_stocks * { 현재 주식 가격}
        self.base_portfolio_value = 0  #직전 학습 시점의 PV
        self.num_buy = 0 # 매수 횟수
        self.num_sell = 0 # 매도 횟수
        self.num_hold = 0 # 광망 횟수
        self.immediate_reward = 0 #즉시 보상

        #Agent 클래스의 상태
        self.ratio_hold = 0 # 주식 보유 비율
        self.ratio_portfolio_value = 0 # 포트폴리오 가치 비율

    #속성 초기화
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    #초기 자본금 설정
    def set_balance(self,balance):
        self.initial_balance = balance

    # 에이전트의 상태를 반환
    def get_states(self):
        #주식 보유 비율 = 주식보유수 / (포트폴리오 가치/ 현재 주가)
        self.ratio_hold = self.num_hold/int(self.portfolio_value / self.environment.get_price())
        # 포트폴리오 가치 비율 = 포트폴리오 가치 / 기준 포트폴리오 가치   0손실 1수익에 가까움
        self.ratio_portfolio_value = self.portfolio_value/self.initial_balance
        return (self.ratio_hold,self.ratio_portfolio_value)

    # 입력으로 들어온 epsilon의 확률로 무작위 행동을 결정 및 정책 신경망 통해 결정
    def decide_action(self,policy_network,sample,epsilon):
        confidence = 0
        # 탐험 결정 무작위 값이 epsilon보다 작으면 무작위로 결정
        if np.random.rand() < epsilon:
            exploration = True
            #무작위로 행동 결정
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            #각 행동에 대한 확률  - 정책 신경망 매수 매도 확률 받아옴
            probs = policy_network.predict(sample)
            action = np.argmax(probs)
            confidence = 1+ probs[action]
        return action,confidence,exploration

    #주식을 사고 팔기전 가능한지 체크
    def validate_action(self,action):
        validity = True
        if action == Agent.ACTION_BUY:
            #가진 돈으로 적어도 1주를 살 수 있는지 결정
            if self.balance < self.environment.get_price()*(1 + self.TRADING_CHARGE)*self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:
            #주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                validity = False

        return validity

    #행동의 확률이 높을수록 매수 또는 매도하는 단위를 크게 결정함
    def decidE_trading_unit(self,confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(int(confidence * (self.max_trading_unit - self.min_trading_unit)),self.max_trading_unit-self.min_trading_unit),0)
        return self.min_trading_unit  + added_trading

    # agent가 결정한 행동을 수행  action 매도 매수 0 1 confidence는 정책 신경망을 통해 결정한 경우 결정 행동의 소프트맥스한 확률값
    def act(self,action,confidence):
        #행동을 할 수 있는지 체크 없는경우 관망
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기 - 주가 받아옴 매수,매도 포트폴리오가치 측정시 사용됨
        curr_price = self.environment.get_price()

        #즉시 보상 초기화
        self.immediate_reward = 0

        #매수
        if action == Agent.ACTION_BUY:
            #매수할 단위를 판단
            trading_unit = self.decidE_trading_unit(confidence)
            balance = self.balance - curr_price*(1 + self.TRADING_CHARGE)*trading_unit
            #보유 현금이 보자랄 경우 보유 현금으로 가능한 만큼 최대한 매수 ex )10개사야되는데 돈이안되는경우 최대한 구입
            if balance < 0:
                trading_unit = max(min(int(self.balance/(curr_price*(1+self.TRADING_CHARGE))),self.max_trading_unit),self.min_trading_unit)
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price*(1+self.TRADING_CHARGE)*trading_unit
            self.balance -= invest_amount #보유 현금을 갱신
            self.num_stocks += trading_unit #보유 주식 수를 갱신
            self.num_buy += 1 #매수 횟수 증가
        #매도
        elif action == Agent.ACTION_SELL:
            #매도할 단위를 판단
            trading_unit = self.decidE_trading_unit(confidence)
            #보유 주식이 모자랄 경우 가능한  만큼 최대한 매도
            trading_unit = min(trading_unit,self.num_stocks)
            #매도
            invest_amount = curr_price*(1-(self.TRADING_TAX + self.TRADING_CHARGE))*trading_unit
            self.num_stocks -= trading_unit # 보유 주식 수를 갱신
            self.balance += invest_amount # 보유 현금을 갱신
            self.num_sell += 1 #매도 횟수 증가
        #관망
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1 #관망 횟수 증가

        #포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price *self.num_stocks
        profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        #즉시 보상 판단
        self.immediate_reward = 1 if profitloss >= 0  else -1

        #지연 보상 판단
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            #목표 수익률을 달성하여 기준 포트폴리오 가치를 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
            #손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward=0

        return self.immediate_reward,delayed_reward


