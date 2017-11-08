__author__ = 'Stephen Lee (mingyangli1314@outlook.com)'

import os, sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

from smart_ml.predictor import Predictor
import numpy as np

import tests.utils_testing as utils

def always_return_ten_thousand(estimator=None, actuals=None, probas=None):
    return 10000


def test_binary_classification():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.get_titanic_binary_classification_dataset()

    column_descriptions = {
        'survived': 'output',
        'sex': 'categorical',
        'embarked': 'categorical',
        'pclass': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier', column_descriptions=column_descriptions)
    ml_predictor.train(df_titanic_train, scoring=always_return_ten_thousand)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print ('test_score')
    print (test_score)

    assert test_score == -10000