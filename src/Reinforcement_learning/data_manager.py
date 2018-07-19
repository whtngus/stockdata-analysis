import pandas as pd
import numpy as np


def load_chart_data(fpath):
    # chart_data = pd.read_csv(fpath, thousands=',', header=None)
    chart_data = pd.read_excel(fpath, error_bad_lines=False)
    # chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    return chart_data[1:]

#종가 5,15,33,56,224,448 데이터 처리
def preprocess(chart_data):
    prep_data = chart_data
    # windows = [5, 10, 20, 60, 120]
    # index number
    windows = [5, 10, 20, 60, 120, 240,480]
    for window in windows:
        # prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['{}거래량'.format(window)] = (
            prep_data['거래량'].rolling(window).mean())
        prep_data[window] =  prep_data["종가"] / prep_data[window]

    return prep_data



def build_training_data(prep_data):
    training_data = prep_data
    # training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
    # training_data.loc[1:, 'open_lastclose_ratio'] = \
    #     (training_data['open'][1:].values - training_data['close'][:-1].values) / \
    #     training_data['close'][:-1].values
    # training_data['high_close_ratio'] = \
    #     (training_data['high'].values - training_data['close'].values) / \
    #     training_data['close'].values
    # training_data['low_close_ratio'] = \
    #     (training_data['low'].values - training_data['close'].values) / \
    #     training_data['close'].values
    # training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    # training_data.loc[1:, 'close_lastclose_ratio'] = \
    #     (training_data['close'][1:].values - training_data['close'][:-1].values) / \
    #     training_data['close'][:-1].values
    # training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    # training_data.loc[1:, 'volume_lastvolume_ratio'] = \
    #     (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
    #     training_data['volume'][:-1]\
    #         .replace(to_replace=0, method='ffill') \
    #         .replace(to_replace=0, method='bfill').values

    windows = [5, 10, 20, 60, 120, 240,480]
    for window in windows:
        training_data['%d거래량' % window] = \
            (training_data['거래량'] - training_data['%d거래량' % window]) / \
            training_data['%d거래량' % window]

    return training_data
