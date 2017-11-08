__author__ = 'Stephen Lee (mingyangli1314@outlook.com)'

from collections import OrderedDict
import os, sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
os.environ['is_test_suite'] = 'True'

import tests.classifiers as classifier_tsets
import tests.regressors as regressor_tests

