import numpy as np
import matplotlib .pyplot as plt
from mpl_finance import candletick_ohlc

class Visualizer:

    def __init__(self):
        #캔버스 역할을 하는 Figure 클래스 객체
        self.fig = None
        # 차트를 그리기 위한 Axes 클래스 객체
        self.axes = None


    def prepare(self,chart_data):
        #캔버스를 초기화하고 4개의 차트를 그림 4행1열 Axes객체로 반환받음
        self.fig,self.axes = plt.subplots(nrows=4,ncols=1,facecolor='w',sharex=True)
        for ax in self.axes:
            #보기 어려운 고학적 표시를 비활성화
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)

        #차트 1. 일봉차트 y축 레이블 표시
        self.axes[0].set_ylabel('Env.')
        #거래량 가시화
        x = np.arange(len(chart_data))
        volume = np.array(chart_data)[:,-1].tolist()
        self.axes[0].bar(x,volume,color='b',alpha=0.3)
        #ohlc란 open,high,low,close
        ax = self.axes[0].twinx()
        ohlc = np.hstack((x.reshape(-1,1),np.array(chart_data)[:,1:-1]))
        #self.axes[0]에 봉 차트 출력
        #양봉은 빨간색 음봉은 파란색 표시
        candletick_ohlc(ax,ohlc,colorup='r',colordown='b')


    def plot(self,epoch_str=None,num_epoches=None,epsilon=None,action_list=None,actions=None,num_stocks=None,outvals=None,exps=None,learning=None,initial_balance=None,pvs=None):
        # epoch_str 제목으로 표시할 에포크
        #모든 차트가 공유할 x축 데이터
        x = np.array(len(actions))
        # 에이전트의 행동 배열
        actions = np.array(action_list)
        #정책 신경망의 출력 배열
        outvals = np.array(outvals)
        #초기 자본금 배열
        pvs_base = np.zeros(len(actions)) + initial_balance

        #차트 2. 에이전트 상태(행동, 보유 주식 수)
        colors = ['r','b']
        for actiontype, color in zip(action_list, colors):
            for i in x[actions == actiontype]:
                # 배경 색으로 행동 표시
                self.axes[1].axvline(i, color=color,alpha = 0.1)
        #보유 주식수 그리기
        self.axes[1].plot(x,num_stocks,'-k')

        #차트 3. 정책 신경망의 출력 및 탐험
        for exp_idx in exps:
            #탐험을 노란색 배경으로 그리기
            self.axes[2].axvline(exp_idx,color='y')
        #탐험하지 않은 지점
        for idx,outval in zip(x,outvals):
            color = 'white'
            if outvals.argmax() == 0:
                # 매수면 빨간색
                color = 'r'
            elif outvals.argmax() == 1:
                # 매도 파란색
                color = 'b'
            #행동을 빨간색 또는 파란색 배경으로 그리기
            self.axes[2].axvline(idx,color=color,alpha=0.1)
        styles = ['.r','.b']
        for action, style in zip(action_list,styles):
            #정책 신경망의 출력을 빤간색, 파란색 점으로 그리기
            self.axes[2].plot(x,outvals[:,action],style)

        #차트 4. 포트폴리오 가치
        #초기자본금 가로로 표시
        self.axes[3].axhline(initial_balance,linestyle='-',color='gray')
        #포트폴리오 가치가 초기 자본금보다 높은 부분 빨간색
        self.axes[3].fill_between(x,pvs,pvs_base,where = pvs> pvs_base, facecolors='r',alpha=0.1)
        #포트폴리오 가치가 초기 자본금보다 낮은 부분 파란색
        self.axes[3].fill_between(x,pvs,pvs_base,where= pvs<pvs_base,facecolors='b',alpha=0.1)
        #포트폴리오 가치 검정색 실선
        self.axes[3].plot(x,pvs,'-k')

        for learning_idx,delayed_reward in learning:
            #학습 위치 표시
            if delayed_reward > 0:
                self.axes[3].axvline(learning_idx,color='r',alpha=0.1)
            else:
                self.axes[3].axvline(learning_idx,color='b',alpha=0.1)

        #에포크 및 탐험 비율
        self.fig.suptitle('Epoch %s/%s (e=%.2f)'%(epoch_str,num_epoches,epsilon))
        #캔버스 레이아웃 조정
        plt.tight_layout()
        plt.subplots_adjust(top=.9)

    #차트 초기화 및 저장 함수 부분
    def clear(self,xlim):
        for ax in self.axes[1:]:
            #그린 차트 지우기
            ax.cla()
            #limit 를 초기화
            ax.relim()
            #스케일 재설정
            ax.autoscale()
        #y축 레이블 재설정
        self.axes[1].set_ylabel('Agent')
        self.axes[2].set_ylabel('PG')
        self.axes[3].set_ylabel('PV')
        for ax in self.axes:
            # x축 limit 재설정
            ax.set_xlim(xlim)
            # 과학적 표기 비활성화
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            # x축 간격을 일정하게 설정
            ax.ticklable_format(useOffset=False)


    def save(self, path):
        plt.savefig(path)
