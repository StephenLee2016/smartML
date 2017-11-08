__author__ = 'Stephen Lee (mingyangli1314@outlook.com)'

import dill
import os
import random
import sys

from smart_ml import utils
from smart_ml import utils_categorical_ensembling
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, \
                                AdaBoostRegressor, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import RANSACRegressor, LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, OrthogonalMatchingPursuit, \
                                BayesianRidge, ARDRegression, SGDClassifier, SGDRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor,\
                                LogisticRegression, RidgeClassifier, Perceptron
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.cluster import MiniBatchKMeans

xgb_installed = False
try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_installed = True
except ImportError:
    pass

lgb_installed = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    lgb_installed = True
except ImportError:
    pass

catboost_installed = False
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    catboost_installed = True
except ImportError:
    pass

keras_imported = False
maxnorm = None
Dense = None
Dropout = None
LeakyReLU = None
ThresholdedReLU = None
PReLU = None
ELU = None
Sequential = None
keras_load_model = None
regularizers = None
optimizers = None
KerasRegressor = None
KerasClassifier = None
Activation = None

# Note: it's important that importing tensorflow come last. We can run into OpenCL issues if we import it ahead of some
# other packages. At the moment, it's a known behavior with tensorflow, but everyone's ok with this workaround.





def get_model_from_name(model_name, training_params=None, is_hp_search=False):
    global keras_imported

    # For Keras
    epochs = 1000
    # if os.environ.get('is_test_suite', 0) == 'True' and model_name[:12] == 'DeepLearning':
    #     print('Heard that this is the test suite. Limiting number of epochs, which will increase training
    #           speed dramatically at the expense of model accuracy')
    #     epochs = 100

    all_model_params = {
        'LogisticRegression': {},
        'RandomForestClassifier': {'n_jobs': -2, 'n_estimators': 30},
        'ExtraTreesClassifier': {'n_jobs': -1},
        'AdaBoostClassifier': {},
        'SGDClassifier': {'n_jobs': -1},
        'Perceptron': {'n_jobs': -1},
        'LinearSVC': {'dual': False},
        'LinearRegression': {'n_jobs': -2},
        'RandomForestRegressor': {'n_jobs': -2, 'n_estimators': 30},
        'LinearSVR': {'dual': False, 'loss': 'squared_epsilon_insensitive'},
        'ExtraTreesRegressor': {'n_jobs': -1},
        'MiniBatchKMeans': {'n_clusters': 8},
        'GradientBoostingRegressor': {'presort': False, 'learning_rate': 0.1, 'warm_start': True},
        'GradientBoostingClassifier': {'presort': False, 'learning_rate': 0.1, 'warm_start': True},
        'SGDRegressor': {'shuffle': False},
        'PassiveAggressiveRegressor': {'shuffle': False},
        'AdaBoostRegressor': {},
        'LGBMRegressor': {'n_estimators': 2000, 'learning_rate': 0.15, 'num_leaves': 8, 'lambda_l2': 0.001},
        'LGBMClassifier': {'n_estimators': 2000, 'learning_rate': 0.15, 'num_leaves': 8, 'lambda_l2': 0.001},
        'DeepLearningRegressor': {'epochs': epochs, 'batch_size': 50, 'verbose': 2},
        'DeepLearningClassifier': {'epochs': epochs, 'batch_size': 50, 'verbose': 2},
        'CatBoostRegressor': {},
        'CatBoostClassifier': {}
    }

    # if os.environ.get('is_test_suite', 0) == 'True':
    #     all_model_params

    model_params = all_model_params.get(model_name, None)
    if model_params is None:
        model_params = {}

    if is_hp_search == True:
        if model_name[:12] == 'DeepLearning':
            model_params['epochs'] = 50
        if model_name[:4] == 'LGBM':
            model_params['n_estimators'] = 500


    if training_params is not None:
        print('Now using the model training_params that you passed in:')
        print(training_params)
        # Overwrite our stock params with what the user passes in (i.e., if the user wants 10,000 trees, we will let them do it)
        model_params.update(training_params)
        print('After overwriting our defaults with your values, here are the final params that will be used to initialize the model:')
        print(model_params)


    model_map = {
        # Classifiers
        'LogisticRegression': LogisticRegression(),
        'RandomForestClassifier': RandomForestClassifier(),
        'RidgeClassifier': RidgeClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),


        'LinearSVC': LinearSVC(),

        # Regressors
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(),
        'Ridge': Ridge(),
        'LinearSVR': LinearSVR(),
        'ExtraTreesRegressor': ExtraTreesRegressor(),
        'AdaBoostRegressor': AdaBoostRegressor(),
        'RANSACRegressor': RANSACRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),

        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'LassoLars': LassoLars(),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'BayesianRidge': BayesianRidge(),
        'ARDRegression': ARDRegression(),

        # Clustering
        'MiniBatchKMeans': MiniBatchKMeans(),
    }

    try:
        model_map['SGDClassifier'] = SGDClassifier(max_iter=1000, tol=0.001)
        model_map['Perceptron'] = Perceptron(max_iter=1000, tol=0.001)
        model_map['PassiveAggressiveClassifier'] = PassiveAggressiveClassifier(max_iter=1000, tol=0.001)
        model_map['SGDRegressor'] = SGDRegressor(max_iter=1000, tol=0.001)
        model_map['PassiveAggressiveRegressor'] = PassiveAggressiveRegressor(max_iter=1000, tol=0.001)
    except TypeError:
        model_map['SGDClassifier'] = SGDClassifier()
        model_map['Perceptron'] = Perceptron()
        model_map['PassiveAggressiveClassifier'] = PassiveAggressiveClassifier()
        model_map['SGDRegressor'] = SGDRegressor()
        model_map['PassiveAggressiveRegressor'] = PassiveAggressiveRegressor()

    if xgb_installed:
        model_map['XGBClassifier'] = XGBClassifier()
        model_map['XGBRegressor'] = XGBRegressor()

    if lgb_installed:
        model_map['LGBMRegressor'] = LGBMRegressor()
        model_map['LGBMClassifier'] = LGBMClassifier()

    if catboost_installed:
        model_map['CatBoostRegressor'] = CatBoostRegressor(calc_feature_importance=True)
        model_map['CatBoostClassifier'] = CatBoostClassifier(calc_feature_importance=True)

    if model_name[:12] == 'DeepLearning':
        if keras_imported == False:
            # Suppress some level of logs if TF is installed (but allow it to not be installed, and use Theano instead)
            try:
                os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                from tensorflow import logging
                logging.set_verbosity(logging.INFO)
            except:
                pass

            global maxnorm
            global Dense, Dropout
            global LeakyReLU, PReLU, ThresholdedReLU, ELU
            global Sequential
            global keras_load_model
            global regularizers, optimizers
            global Activation
            global KerasRegressor, KerasClassifier

            from keras.constraints import maxnorm
            from keras.layers import Activation, Dense, Dropout
            from keras.layers.advanced_activations import LeakyReLU, PReLU, ThresholdedReLU, ELU
            from keras.models import Sequential
            from keras.models import load_model as keras_load_model
            from keras import regularizers, optimizers
            from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
            keras_imported = True

        model_map['DeepLearningClassifier'] = KerasClassifier(build_fn=make_deep_learning_classifier)
        model_map['DeepLearningRegressor'] = KerasRegressor(build_fn=make_deep_learning_model)

    try:
        model_without_params = model_map[model_name]
    except KeyError as e:
        print('It appears you are trying to use a library that is not available when we try to import it, '
              'or using a value for model_names that we do not recognize')
        raise(e)

    if os.environ.get('is_test_suite', False) == 'True':
        if 'n_jobs' in model_params:
            model_params['n_jobs'] = 1
    model_with_params = model_without_params.set_params(**model_params)

    return model_with_params


def get_name_from_model(model):
    if isinstance(model, LogisticRegression):
        return 'LogisticRegression'
    if isinstance(model, RandomForestClassifier):
        return 'RandomForestClassifier'
    if isinstance(model, RidgeClassifier):
        return 'RidgeClassifier'
    if isinstance(model, GradientBoostingClassifier):
        return 'GradientBoostingClassifier'
    if isinstance(model, ExtraTreesClassifier):
        return 'ExtraTreesClassifier'
    if isinstance(model, AdaBoostClassifier):
        return 'AdaBoostClassifier'
    if isinstance(model, SGDClassifier):
        return 'SGDClassifier'
    if isinstance(model, Perceptron):
        return 'Perceptron'
    if isinstance(model, PassiveAggressiveClassifier):
        return 'PassiveAggressiveClassifier'
    if isinstance(model, LinearRegression):
        return 'LinearRegression'
    if isinstance(model, RandomForestRegressor):
        return 'RandomForestRegressor'
    if isinstance(model, Ridge):
        return 'Ridge'
    if isinstance(model, ExtraTreesRegressor):
        return 'ExtraTreesRegressor'
    if isinstance(model, AdaBoostRegressor):
        return 'AdaBoostRegressor'
    if isinstance(model, RANSACRegressor):
        return 'RANSACRegressor'
    if isinstance(model, GradientBoostingRegressor):
        return 'GradientBoostingRegressor'
    if isinstance(model, Lasso):
        return 'Lasso'
    if isinstance(model, ElasticNet):
        return 'ElasticNet'
    if isinstance(model, LassoLars):
        return 'LassoLars'
    if isinstance(model, OrthogonalMatchingPursuit):
        return 'OrthogonalMatchingPursuit'
    if isinstance(model, BayesianRidge):
        return 'BayesianRidge'
    if isinstance(model, ARDRegression):
        return 'ARDRegression'
    if isinstance(model, SGDRegressor):
        return 'SGDRegressor'
    if isinstance(model, PassiveAggressiveRegressor):
        return 'PassiveAggressiveRegressor'
    if isinstance(model, MiniBatchKMeans):
        return 'MiniBatchKMeans'
    if isinstance(model, LinearSVR):
        return 'LinearSVR'
    if isinstance(model, LinearSVC):
        return 'LinearSVC'

    if xgb_installed:
        if isinstance(model, XGBClassifier):
            return 'XGBClassifier'
        if isinstance(model, XGBRegressor):
            return 'XGBRegressor'

    if keras_imported:
        if isinstance(model, KerasRegressor):
            return 'DeepLearningRegressor'
        if isinstance(model, KerasClassifier):
            return 'DeepLearningClassifier'

    if lgb_installed:
        if isinstance(model, LGBMClassifier):
            return 'LGBMClassifier'
        if isinstance(model, LGBMRegressor):
            return 'LGBMRegressor'

    if catboost_installed:
        if isinstance(model, CatBoostClassifier):
            return 'CatBoostClassifier'
        if isinstance(model, CatBoostRegressor):
            return 'CatBoostRegressor'

# Hyperparameter search spaces for each model
def get_search_params(model_name):
    grid_search_params = {
        'DeepLearningRegressor': {
            'hidden_layers': [
                [1],
                [1, 0.1],
                [1, 1, 1],
                [1, 0.5, 0.1],
                [2],
                [5],
                [1, 0.5, 0.25, 0.1, 0.05],
                [1, 1, 1, 1],
                [1, 1]

                # [1],
                # [0.5],
                # [2],
                # [1, 1],
                # [0.5, 0.5],
                # [2, 2],
                # [1, 1, 1],
                # [1, 0.5, 0.5],
                # [0.5, 1, 1],
                # [1, 0.5, 0.25],
                # [1, 2, 1],
                # [1, 1, 1, 1],
                # [1, 0.66, 0.33, 0.1],
                # [1, 2, 2, 1]
            ]
            , 'dropout_rate': [0.0, 0.2, 0.4, 0.6, 0.8]
            , 'kernel_initializer': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
                                     'glorot_uniform', 'he_normal', 'he_uniform']
            , 'activation': ['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid',
                             'hard_sigmoid', 'linear', 'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU']
            , 'batch_size': [16, 32, 64, 128, 256, 512]
            , 'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        },
        'DeepLearningClassifier': {
            'hidden_layers': [
                [1],
                [0.5],
                [2],
                [1, 1],
                [0.5, 0.5],
                [2, 2],
                [1, 1, 1],
                [1, 0.5, 0.5],
                [0.5, 1, 1],
                [1, 0.5, 0.25],
                [1, 2, 1],
                [1, 1, 1, 1],
                [1, 0.66, 0.33, 0.1],
                [1, 2, 2, 1]
            ]
            , 'batch_size': [16, 32, 64, 128, 256, 512]
            , 'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
            , 'activation': ['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', 'linear',
                             'LeakyReLU', 'PReLU', 'ELU', 'ThresholdedReLU']
            # , 'epochs': [2, 4, 6, 10, 20]
            # , 'batch_size': [10, 25, 50, 100, 200, 1000]
            # , 'lr': [0.001, 0.01, 0.1, 0.3]
            # , 'momentum': [0.0, 0.3, 0.6, 0.8, 0.9]
            # , 'init_mode': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
            # , 'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
            # , 'weight_constraint': [1, 3, 5]
            , 'dropout_rate': [0.0, 0.3, 0.6, 0.8, 0.9]
        },
        'XGBClassifier': {
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 75, 100, 150, 200, 375, 500, 750, 1000],
            'min_child_weight': [1, 5, 10, 50],
            'subsample': [0.5, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.8, 1.0]
            # 'subsample': [0.5, 1.0]
            # 'lambda': [0.9, 1.0]
        },
        'XGBRegressor': {
            # Add in max_delta_step if classes are extremely imbalanced
            'max_depth': [1, 3, 8, 25],
            # 'lossl': ['ls', 'lad', 'huber', 'quantile'],
            # 'booster': ['gbtree', 'gblinear', 'dart'],
            # 'objective': ['reg:linear', 'reg:gamma'],
            # 'learning_rate': [0.01, 0.1],
            'subsample': [0.5, 1.0]
            # 'subsample': [0.4, 0.5, 0.58, 0.63, 0.68, 0.76],

        },
        'GradientBoostingRegressor': {
            # Add in max_delta_step if classes are extremely imbalanced
            'max_depth': [1, 2, 3, 4, 5, 7, 10, 15],
            'max_features': ['sqrt', 'log2', None],
            'loss': ['ls', 'huber'],
            'learning_rate': [0.001, 0.01, 0.05,  0.1, 0.2],
            'n_estimators': [10, 50, 75, 100, 125, 150, 200, 500, 1000, 2000],
            'subsample': [0.5, 0.65, 0.8, 0.9, 0.95, 1.0]
        },
        'GradientBoostingClassifier': {
            'loss': ['deviance', 'exponential'],
            'max_depth': [1, 2, 3, 4, 5, 7, 10, 15],
            'max_features': ['sqrt', 'log2', None],
            'learning_rate': [0.001, 0.01, 0.05,  0.1, 0.2],
            'subsample': [0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
            'n_estimators': [10, 50, 75, 100, 125, 150, 200, 500, 1000, 2000],

        },

        'LogisticRegression': {
            'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
            'class_weight': [None, 'balanced'],
            'solver': ['newton-cg', 'lbfgs', 'sag']
        },
        'LinearRegression': {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        },
        'RandomForestClassifier': {
            'criterion': ['entropy', 'gini'],
            'class_weight': [None, 'balanced'],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': [2, 5, 20, 50, 100],
            'min_samples_leaf': [1, 2, 5, 20, 50, 100],
            'bootstrap': [True, False]
        },
        'RandomForestRegressor': {
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_split': [2, 5, 20, 50, 100],
            'min_samples_leaf': [1, 2, 5, 20, 50, 100],
            'bootstrap': [True, False]
        },
        'RidgeClassifier': {
            'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
            'class_weight': [None, 'balanced'],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
        },
        'Ridge': {
            'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
        },
        'ExtraTreesRegressor': {
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_split': [2, 5, 20, 50, 100],
            'min_samples_leaf': [1, 2, 5, 20, 50, 100],
            'bootstrap': [True, False]
        },
        'AdaBoostRegressor': {
            'base_estimator': [None, LinearRegression(n_jobs=-1)],
            'loss': ['linear','square','exponential']
        },
        'RANSACRegressor': {
            'min_samples': [None, .1, 100, 1000, 10000],
            'stop_probability': [0.99, 0.98, 0.95, 0.90]
        },
        'Lasso': {
            'selection': ['cyclic', 'random'],
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'positive': [True, False]
        },

        'ElasticNet': {
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'selection': ['cyclic', 'random'],
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'positive': [True, False]
        },

        'LassoLars': {
            'positive': [True, False],
            'max_iter': [50, 100, 250, 500, 1000]
        },

        'OrthogonalMatchingPursuit': {
            'n_nonzero_coefs': [None, 3, 5, 10, 25, 50, 75, 100, 200, 500]
        },

        'BayesianRidge': {
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'alpha_1': [.0000001, .000001, .00001, .0001, .001],
            'lambda_1': [.0000001, .000001, .00001, .0001, .001],
            'lambda_2': [.0000001, .000001, .00001, .0001, .001]
        },

        'ARDRegression': {
            'tol': [.0000001, .000001, .00001, .0001, .001],
            'alpha_1': [.0000001, .000001, .00001, .0001, .001],
            'alpha_2': [.0000001, .000001, .00001, .0001, .001],
            'lambda_1': [.0000001, .000001, .00001, .0001, .001],
            'lambda_2': [.0000001, .000001, .00001, .0001, .001],
            'threshold_lambda': [100, 1000, 10000, 100000, 1000000]
        },

        'SGDRegressor': {
            'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'alpha': [.0000001, .000001, .00001, .0001, .001]
        },

        'PassiveAggressiveRegressor': {
            'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5],
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
        },

        'SGDClassifier': {
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber',
                     'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'alpha': [.0000001, .000001, .00001, .0001, .001],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'class_weight': ['balanced', None]
        },

        'Perceptron': {
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'alpha': [.0000001, .000001, .00001, .0001, .001],
            'class_weight': ['balanced', None]
        },

        'PassiveAggressiveClassifier': {
            'loss': ['hinge', 'squared_hinge'],
            'class_weight': ['balanced', None],
            'C': [0.01, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        }

        , 'LGBMClassifier': {
            'boosting_type': ['gbdt', 'dart']
            , 'max_bin': [25, 50, 100, 200, 250, 300, 400, 500, 750, 1000]
            , 'min_child_samples': [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000]
            , 'num_leaves': [2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250]
            , 'colsample_bytree': [0.7, 0.9, 1.0]
            , 'subsample': [0.7, 0.9, 1.0]
            , 'learning_rate': [0.01, 0.05, 0.1]
            , 'n_estimators': [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000]
        }

        , 'LGBMRegressor': {
            'boosting_type': ['gbdt', 'dart']
            , 'max_bin': [25, 50, 100, 200, 250, 300, 400, 500, 750, 1000]
            , 'min_child_samples': [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000]
            , 'num_leaves': [2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250]
            , 'colsample_bytree': [0.7, 0.9, 1.0]
            , 'subsample': [0.7, 0.9, 1.0]
            , 'learning_rate': [0.01, 0.05, 0.1]
            , 'n_estimators': [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000]
        }

        , 'CatBoostClassifier': {
            'depth': [1, 2, 3, 5, 7, 9, 12, 15, 20, 32]
            , 'l2_leaf_reg': [.0000001, .000001, .00001, .0001, .001, .01, .1]
            , 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]

            # , random_strength
            # , bagging_temperature
        }

        , 'CatBoostRegressor': {
            'depth': [1, 2, 3, 5, 7, 9, 12, 15, 20, 32]
            , 'l2_leaf_reg': [.0000001, .000001, .00001, .0001, .001, .01, .1]
            , 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]

            # , random_strength
            # , bagging_temperature
        }

        , 'LinearSVR': {
            'C': [0.5, 0.75, 0.85, 0.95, 1.0]
            , 'epsilon': [0, 0.05, 0.1, 0.15, 0.2]
        }

        , 'LinearSVC': {
            'C': [0.5, 0.75, 0.85, 0.95, 1.0]
        }

    }

    # Some of these are super expensive to compute. So if we're running this in a test suite, let's make sure the structure works,
    # but reduce the compute time
    params = grid_search_params[model_name]
    if os.environ.get('is_test_suite', 0) == 'True' and model_name[:8] == 'CatBoost':
        simplified_params = {}
        for k, v in params.items():
            # Grab the first two items for each thing we want to test
            simplified_params[k] = v[:2]
        params = simplified_params

    return params


def insert_deep_learning_model(pipeline_step, file_name):
    # This is where we saved the random_name for this model
    random_name = pipeline_step.model
    # Load the Keras model here
    keras_file_name = file_name[:-5] + random_name + '_keras_deep_learning_model.h5'

    model = keras_load_model(keras_file_name)

    # Put the model back in place so that we can still use it to get predictions without having to load it back in from disk
    return model

def load_ml_model(file_name):

    with open(file_name, 'rb') as read_file:
        base_pipeline = dill.load(read_file)

    if isinstance(base_pipeline, utils_categorical_ensembling.CategoricalEnsembler):
        for step in base_pipeline.transformation_pipeline.named_steps:
            pipeline_step = base_pipeline.transformation_pipeline.named_steps[step]

            try:
                if pipeline_step.get('model_name', 'reallylongnonsensicalstring')[:12] == 'DeepLearning':
                    pipeline_step.model = insert_deep_learning_model(pipeline_step, file_name)
            except AttributeError:
                pass

        for step in base_pipeline.trained_models:
            pipeline_step = base_pipeline.trained_models[step]

            try:
                if pipeline_step.get('model_name', 'reallylongnonsensicalstring')[:12] == 'DeepLearning':
                    pipeline_step.model = insert_deep_learning_model(pipeline_step, file_name)
            except AttributeError:
                pass

    else:

        for step in base_pipeline.named_steps:
            pipeline_step = base_pipeline.named_steps[step]
            try:
                if pipeline_step.get('model_name', 'reallylongnonsensicalstring')[:12] == 'DeepLearning':
                    pipeline_step.model = insert_deep_learning_model(pipeline_step, file_name)
            except AttributeError:
                pass

    return base_pipeline

# Keeping this here for legacy support
def load_keras_model(file_name):
    return load_ml_model(file_name)

# For many activations, we can just pass the activation name into Activations
# For some others, we have to import them as their own standalone activation function
def get_activation_layer(activation):
    if activation == 'LeakyReLU':
        return LeakyReLU()
    if activation == 'PReLU':
        return PReLU()
    if activation == 'ELU':
        return ELU()
    if activation == 'ThresholdedReLU':
        return ThresholdedReLU()

    return Activation(activation)

# TODO: same for optimizers, including clipnorm
def get_optimizer(name='Adadelta'):
    if name == 'SGD':
        return optimizers.SGD(clipnorm=1.)
    if name == 'RMSprop':
        return optimizers.RMSprop(clipnorm=1.)
    if name == 'Adagrad':
        return optimizers.Adagrad(clipnorm=1.)
    if name == 'Adadelta':
        return optimizers.Adadelta(clipnorm=1.)
    if name == 'Adam':
        return optimizers.Adam(clipnorm=1.)
    if name == 'Adamax':
        return optimizers.Adamax(clipnorm=1.)
    if name == 'Nadam':
        return optimizers.Nadam(clipnorm=1.)

    return optimizers.Adam(clipnorm=1.)



def make_deep_learning_model(hidden_layers=None, num_cols=None, optimizer='Adadelta', dropout_rate=0.2, weight_constraint=0,
                             feature_learning=False, kernel_initializer='normal', activation='elu'):

    if feature_learning == True and hidden_layers is None:
        hidden_layers = [1, 0.75, 0.25]

    if hidden_layers is None:
        hidden_layers = [1, 0.75, 0.25]

    # The hidden_layers passed to us is simply describing a shape. it does not know the num_cols we are dealing with,
    # it is simply values of 0.5, 1, and 2, which need to be multiplied by the num_cols
    scaled_layers = []
    for layer in hidden_layers:
        scaled_layers.append(min(int(num_cols * layer), 10))

    # If we're training this model for feature_learning, our penultimate layer (our final hidden layer before the "output" layer)
    # will always have 10 neurons, meaning that we always output 10 features from our feature_learning model
    if feature_learning == True:
        scaled_layers.append(10)

    model = Sequential()

    model.add(Dense(scaled_layers[0], input_dim=num_cols, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
    model.add(get_activation_layer(activation))

    for layer_size in scaled_layers[1:-1]:
        model.add(Dense(layer_size, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
        model.add(get_activation_layer(activation))

    # There are times we will want the output from our penultimate layer, not the final layer, so give it a name that makes
    # the penultimate layer easy to find
    model.add(Dense(scaled_layers[-1], kernel_initializer=kernel_initializer, name='penultimate_layer', kernel_regularizer=regularizers.l2(0.01)))
    model.add(get_activation_layer(activation))

    # For regressors, we want an output layer with a single node
    model.add(Dense(1, kernel_initializer=kernel_initializer))


    # The final step is to compile the model
    model.compile(loss='mean_squared_error', optimizer=get_optimizer(optimizer), metrics=['mean_absolute_error',
                                                                                          'mean_absolute_percentage_error'])

    return model


def make_deep_learning_classifier(hidden_layers=None, num_cols=None, optimizer='Adadelta', dropout_rate=0.2,
                                  weight_constraint=0, final_activation='sigmoid', feature_learning=False,
                                  activation='elu', kernel_initializer='normal'):

    if feature_learning == True and hidden_layers is None:
        hidden_layers = [1, 0.75, 0.25]

    if hidden_layers is None:
        hidden_layers = [1, 0.75, 0.25]

    # The hidden_layers passed to us is simply describing a shape. it does not know the num_cols we are dealing with,
    # it is simply values of 0.5, 1, and 2, which need to be multiplied by the num_cols
    scaled_layers = []
    for layer in hidden_layers:
        scaled_layers.append(min(int(num_cols * layer), 10))

    # If we're training this model for feature_learning, our penultimate layer (our final hidden layer before the "output" layer)
    # will always have 10 neurons, meaning that we always output 10 features from our feature_learning model
    if feature_learning == True:
        scaled_layers.append(10)


    model = Sequential()

    # There are times we will want the output from our penultimate layer, not the final layer, so give it a name that makes
    # the penultimate layer easy to find
    model.add(Dense(scaled_layers[0], input_dim=num_cols, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
    model.add(get_activation_layer(activation))

    for layer_size in scaled_layers[1:-1]:
        model.add(Dense(layer_size, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
        model.add(get_activation_layer(activation))

    model.add(Dense(scaled_layers[-1], kernel_initializer=kernel_initializer, name='penultimate_layer', kernel_regularizer=regularizers.l2(0.01)))
    model.add(get_activation_layer(activation))

    model.add(Dense(1, kernel_initializer=kernel_initializer, activation=final_activation))
    model.compile(loss='binary_crossentropy', optimizer=get_optimizer(optimizer), metrics=['accuracy', 'poisson'])
    return model

