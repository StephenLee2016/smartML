__author__ = 'Stephen Lee (mingyangli1314@outlook.com)'

import datetime
import os, sys
import random
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
os.environ['is_test_suite'] = 'True'

from smart_ml.predictor import Predictor
from smart_ml.utils_models import load_ml_model

from nose.tools import assert_equal, assert_not_equal, with_setup
from sklearn.metrics import accuracy_score

import dill
import numpy as np
import tests.utils_testing as utils


def ensemble_classifier_basic_test(model_name=None):
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()
    column_descriptions = {
        'survived': 'output',
        'sex': 'categorical',
        'embarked': 'categorical',
        'pclass': 'categorical'
    }

    ensemble_config = [
        {
            'model_name': 'LGBMClassifier'
        },

        {
            'model_name': 'RandomForestClassifier'
        }
    ]

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train, ensemble_config=ensemble_config)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print ('test_score')
    print (test_score)

    assert -0.15 < test_score < -0.131


def ensemble_regressor_basic_test():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output',
        'CHAS': 'categorical'
    }

    ensemble_config = [
        {
            'model_name': 'LGBMRegressor'
        },

        {
            'model_name': 'RandomForestRegressor'
        }
    ]


    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
    ml_predictor.train(df_boston_train, ensemble_config=ensemble_config)
    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print ('test_score')
    print (test_score)

    assert -3.0 < test_score < -2.8


# TODO: test for warning when passing in ensemble_method!='average' and is_classifier
# TODO: make sure this works for single predictions and batch
def getting_single_predictions_classifier_test():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output',
        'sex': 'categorical',
        'embarked': 'categorical',
        'pclass': 'categorical',
        'age_bucket': 'categorical'
    }

    ensemble_config = [
        {
            'model_name': 'LGBMClassifier'
        },

        {
            'model_name': 'RandomForestClassifier'
        }
    ]

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train, ensemble_config=ensemble_config)

    file_name = ml_predictor.save(str(random.random()))
    saved_ml_pipeline = load_ml_model(file_name)

    os.remove(file_name)
    try:
        kerea_file_name = file_name[:-5] + '_keras_deep_learning_model.h5'
        os.remove(kerea_file_name)
    except:
        pass

    df_titanic_test_dictionaries = df_titanic_test.to_dict('records')


    # 1. Make sure the accuracy is the same
    predictions = []
    for row in df_titanic_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict_proba(row)[1])

    print ('predictions')
    print (predictions)

    first_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
    print ('first_score')
    print (first_score)

    # Make sure our score is good, but not unreasonably good
    lower_bound = -0.16

    assert -0.15 < first_score < -0.135

    # 2. Make sure the speed is reasonable (do it a few extra times)
    data_length = len(df_titanic_test_dictionaries)
    start_time = datetime.datetime.now()
    for idx in range(1000):
        row_num = idx % data_length
        saved_ml_pipeline.predict(df_titanic_test_dictionaries[row_num])
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print ('duration.total_seconds()')
    print (duration.total_seconds())

    assert 0.2 < duration.total_seconds() < 60


    # 3. Make sure we're not modifying the dictionaries ( the score is the same after running a few experiments as it is the first time)
    predictions = []
    for row in df_titanic_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict_proba(row)[1])

    print ('predictions')
    print (predictions)
    print ('df_titanic_test_dictionaries')
    print (df_titanic_test_dictionaries)
    second_score = utils.calculate_brier_score_loss(df_titanic_test.survived, predictions)
    print ('second_score')
    print (second_score)

    # Make sure our score is good, but not unreasonably good
    assert -0.15 < second_score < -0.135


def getting_single_predictions_regressor_test():
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output',
        'CHAS': 'categorical'
    }

    ensemble_config = [
        {
            'model_name': 'LGBMRegressor'
        },

        {
            'model_name': 'RandomForestRegressor'
        }
    ]

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    # NOTE: this is bad practice to pass in our same training set as out fl_data set , but we don't have enough data to do it any other way
    ml_predictor.train(df_boston_train, ensemble_config=ensemble_config)
    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print ('test_score')
    print (test_score)

    assert -3.5 < test_score < -2.8

    file_name = ml_predictor.save(str(random.random()))
    saved_ml_pipeline = load_ml_model(file_name)

    os.remove(file_name)
    try:
        keras_file_name = file_name[:-5] + '_keras_deep_learning_model.h5'
        os.remove(keras_file_name)
    except:
        pass

    df_boston_test_dictionaries = df_boston_test.to_dict('records')

    # 1. Make sure the accuracy is the same
    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    first_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print ('first_score')
    print (first_score)

    # make sure our score is good, but not unreasonably good
    lower_bound = -3.5
    assert lower_bound < first_score < -2.8

    # 2. Make sure the speed is reasonable (do it a few extra times)
    data_length = len(df_boston_test_dictionaries)
    start_time = datetime.datetime.now()
    for idx in range(1000):
        row_num = idx % data_length
        saved_ml_pipeline.predict(df_boston_test_dictionaries[row_num])
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print ('duration.total_seconds()')
    print (duration.total_seconds())

    assert 0.2 < duration.total_seconds() / 1.0 < 60

    # 3. Make sure we're not modifying the dictionaries (the score is the same after running a few experiments as it is the first time)
    predictions = []
    for row in df_boston_test_dictionaries:
        predictions.append(saved_ml_pipeline.predict(row))

    second_score = utils.calculate_rmse(df_boston_test.MEDV, predictions)
    print ('second_score')
    print (second_score)

    # Make sure our score is good, but not unreasonably good
    assert lower_bound < second_score < -2.8


