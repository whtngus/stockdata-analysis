import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner


if __name__ == '__main__':
    stock_code = 'chobi.xlsx'  # 삼성전자

    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    timestr = settings.get_time_str()

    if not os.path.exists('logs/%s' % stock_code):
        os.makedirs('logs/%s' % stock_code)

    file_handler = logging.FileHandler(filename=os.path.join(log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 주식 데이터 준비
    chart_data = data_manager.load_chart_data(
        os.path.join(settings.BASE_DIR,
                     'data/kiwoom/{}'.format(stock_code)))
    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)
    #모든기간 사용으로 아래내용 주석처리
    # 기간 필터링
    # training_data = training_data[(training_data['date'] >= '2017-01-01') &
    #                               (training_data['date'] <= '2017-12-31')]

    training_data = training_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['날짜','종가', '거래량']
    chart_data = training_data[features_chart_data]
    # 학습 데이터 분리
    features_training_data = [
        '종가',5,15,33,56,224,448,'전환선 9','기준선 26','후행스팬 26','선행스팬1 9,26','선행스팬2 52,26',
        'Bollinger Bands중심선 20,2.4','상한선','하한선','Para SAR 강세패턴','약세패턴','골든크로스','데드크로스',
        '##5이평 15일선 골든크로스신호','##5이평 20일선 골든크로스신호','거래량',
        '5거래량','15거래량','33거래량','56거래량','224거래량','448거래량','OBV','Signal 9',
        'VR 21','Slow %K 20,5','Slow %D 3','RSI 21','CCI 42'

    ]
    training_data = training_data[features_training_data]

    # 강화학습 시작
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.2, lr=.001)
    policy_learner.fit(balance=10000000, num_epoches=1000,
                       discount_factor=0, start_epsilon=.5)

    # 정책 신경망을 파일로 저장
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)