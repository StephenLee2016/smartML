__author__ = 'Stephen Lee (mingyangli1314@outlook.com)'

'''
nosetests -sv --nologcapture tests/quick_test.py
nosetests --verbosity=2 --detailed-errors --nologcapture --processes=4 --process-restartworker --process-timeout=1000 tests/quick_test.py
'''

import datetime
import os, sys
import random
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
os.environ['is_test_suite'] = 'True'
os.environ['KERAS_BACKEND'] = 'tensorflow'

from smart_ml.predictor import Predictor
from smart_ml.utils_models import load_ml_model

from nose.tools import assert_not_equal, assert_equal, with_setup
from sklearn.metrics import accuracy_score

import dill
import numpy as np
import tests.utils_testing as utils



def regression_test():
    # a random seed of 42 has ExtraTreesRegressor getting the best CV score, and that model doesn't generalize as well as GradientBoostingRegressor.
    np.random.seed(0)
    model_name = 'DeepLearningRegressor'

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output',
        'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)
    ml_predictor.train(df_boston_train, model_names=['DeepLearningRegressor'])

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print ('test_score')
    print (test_score)

    lower_bound = -3.2
    if model_name == 'DeepLearningRegressor':
        lower_bound = -7.8
    if model_name == 'LGBMRegressor':
        lower_bound = -4.95
    if model_name == 'XGBRegressor':
        lower_bound = -3.4

    assert lower_bound < test_score < -2.8


def classification_test():
    np.random.seed(0)
    model_name = ['DeepLearningClassifier', 'XGBClassifier']

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output',
        'embarked': 'categorical',
        'pclass': 'categorical',
        'name': 'nlp',
        'home.dest': 'nlp'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train, model_names=model_name)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print ('test_score')
    print (test_score)

    lower_bound = -0.215
    if model_name == 'DeepLearningClassifier':
        lower_bound = -0.245
    if model_name == 'LGBMClassifier':
        lower_bound = -0.225

    assert lower_bound < test_score < -0.17