__author__ = 'Stephen Lee (mingyangli1314@outlook.com)'

import os, sys
sys.path = [os.path.abspath((os.path.dirname(__file__)))] + sys.path

from smart_ml.predictor import Predictor

import dill
from nose.tools import assert_equal, assert_not_equal, with_setup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tests.utils_testing as utils


def test_predict_uncertainty_returns_pandas_DataFrame_for_more_than_one_values():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output',
        'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)
    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=True, train_uncertainty_model=True, uncertainty_data=uncertainty_data)
    uncertainties = ml_predictor.predict_uncertainty(df_boston_test)

    assert isinstance(uncertainties, pd.DataFrame)


def test_predict_uncertainty_returns_dict_for_one_value():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output',
        'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)
    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=True, train_uncertainty_model=True, uncertainty_data=uncertainty_data)

    test_list = df_boston_test.to_dict('records')

    for item in test_list:
        prediction = ml_predictor.predict_uncertainty(item)
        assert isinstance(prediction, dict)



def test_score_uncertainty():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output',
        'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)
    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train(df_boston_train, perform_feature_selection=True, train_uncertainty_model=True, uncertainty_data=uncertainty_data)

    uncertainty_score = ml_predictor.score_uncertainty(df_boston_test, df_boston_test.MEDV)

    print ('uncertainty_score')
    print (uncertainty_score)

    assert uncertainty_score > -0.2



def test_calibrate_uncertainty():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output',
        'CHAS': 'categorical'
    }

    df_boston_train, uncertainty_data = train_test_split(df_boston_train, test_size=0.5)
    uncertainty_data, uncertainty_calibration_data = train_test_split(uncertainty_data, test_size=0.5)

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    uncertainty_calibration_settings = {
        'num_buckets': 3,
        'percentiles': [25, 50, 75]
    }
    ml_predictor.train(df_boston_train, perform_feature_selection=True, train_uncertainty_model=True, uncertainty_data=uncertainty_data,
                       calibrate_uncertainty=True, uncertainty_calibration_settings=uncertainty_calibration_settings,
                       uncertainty_calibration_data=uncertainty_calibration_data)

    uncertainty_score = ml_predictor.predict_uncertainty(df_boston_test)

    assert 'percentile_25_delta' in list(uncertainty_score.columns)
    assert 'percentile_50_delta' in list(uncertainty_score.columns)
    assert 'percentile_75_delta' in list(uncertainty_score.columns)
    assert 'bucket_num' in list(uncertainty_score.columns)
