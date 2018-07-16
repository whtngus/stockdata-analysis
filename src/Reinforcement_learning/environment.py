class Environment:
	#종가 위치 - 데이터에 맞게 적기
	PRICE_IDX =1
	#차트 데이터를 할당 받음 
	def __init__(self,chart_data=None):
		self.chart_data = chart_data
		print("chart_data : ",self.chart_data.iloc[self.idx])
		self.observation = None
		self.idx = -1

    #차트 데이터의 처음으로 돌아가기
	def reset(self):
		self.observation = None
		self.idx = -1

    #하루 앞으로 이동 및 관측 데이터 제공
	def observe(self):
		if len(self.chart_data) > self.idx+1:
			self.idx += 1
            #iloc 특정 행의 데이터를 가져옴
			self.observation = self.chart_data.iloc[self.idx]
			return self.observation
		return None

    #관측 데이터로부터 종가 반환
	def get_price(self):
		if self.observation is not None:
			return self.observation[self.PRICE_IDX]
		return None
