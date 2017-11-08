__author__ = 'Stephen Lee (mingyangli1314@outlook.com)'

import csv
import datetime
import os
import codecs

import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import column_or_1d

def write_gs_param_results_to_file(trained_gs, most_recent_filename):
    '''
    将 grid search 的结果写入文件
    :param trained_gs:
    :param most_recent_filename:
    :return:
    '''
    timestamp_time = datetime.datetime.now()
    write_most_recent_gs_result_to_file(trained_gs, most_recent_filename, timestamp_time)

    grid_scores = trained_gs.grid_scores_
    scorer = trained_gs.scorer_
    best_score = trained_gs.best_score_

    file_name = 'pipeline_grid_search_results.csv'
    write_header = False
    if not os.path.isfile(file_name):
        write_header = True

    with codecs.open(file_name, 'a', 'utf-8') as results_file:
        writer = csv.writer(results_file, dialect='excel')
        if write_header:
            writer.writerow(['timestamp', 'scorer', 'best_score', 'all_grid_scores'])
        writer.writerow([timestamp_time, scorer, best_score, grid_scores])


def write_most_recent_gs_result_to_file(trained_gs, most_recent_filename, timestamp):
    '''

    :param trained_gs:
    :param most_recent_filename:
    :param timestamp:
    :return:
    '''
    timestamp_time = timestamp
    grid_scores = trained_gs.grid_scores_
    scorer = trained_gs.scorer_
    best_score = trained_gs.best_score_

    file_name = most_recent_filename

    write_header = False
    make_header = False
    if not os.path.isfile(most_recent_filename):
        header_row = ['timestamp', 'scorer', 'best_score', 'cv_mean', 'cv_all']
        write_header = True
        make_header = True

    rows_to_write = []

    for score in grid_scores:
        row = [timestamp_time, scorer, best_score, score[1], score[2]]

        for k, v in score[0].items():
            if make_header:
                header_row.append(k)
            row.append(v)
        rows_to_write.append(row)
        make_header = False

    with codecs.open(file_name, 'a', 'utf-8') as results_file:
        writer = csv.writer(results_file, dialect='excel')
        if write_header:
            writer.writerow(header_row)
        for row in rows_to_write:
            writer.writerow(row)

def safely_drop_columns(df, cols_to_drop):
    '''
    删掉dataframe里要删除的列
    :param df:
    :param cols_to_drop:
    :return:
    '''
    safe_cols_to_drop = []
    for col in cols_to_drop:
        if col in df.columns:
            safe_cols_to_drop.append(col)

    df = df.drop(safe_cols_to_drop, axis=1)
    return df

def drop_duplicate_columns(df):
    '''
    删掉重复的列
    :param df:
    :return:
    '''
    cols = list(df.columns)
    for idx, item in enumerate(df.columns):
        if item in df.columns[:idx]:
            print ("#"*30)
            print ('A duplicate column has be found, and it will be removed now')
            print ('If you intended to send in two different pieces of information, please make sure '\
                   'they have different column names')
            print ('Here is the duplicate column: ')
            print (item)
            print ('#'*30)
            cols[idx] = 'toDROP'
    df.columns = cols

    try:
        df = df.drop('toDROP', axis=1)
    except:
        pass
    return df

def get_boston_dataset():
    boston = load_boston()
    df_boston = pd.DataFrame(boston.data)
    df_boston.columns = boston.feature_names
    df_boston['MEDV'] = boston['target']
    df_boston_train, df_boston_test = train_test_split(df_boston, test_size=0.2, random_state=42)
    return df_boston_train, df_boston_test

bad_vals_as_strings = set([str(float('nan')), str(float('inf')), str(float('-inf')), 'None', 'none', 'NaN', 'NAN', 'nan', 'NULL', 'null', '', 'inf', '-inf'])

def delete_rows_csr(mat, indices):
    '''
    Remove the rows denoted by ''indeices'' from the CSR sparse matrix ''mat''
    :param mat:
    :param indices:
    :return:
    '''
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError('works only for CSR format -- use .tocsr() first')
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def drop_missing_y_vals(df, y, output_column=None):
    '''
    删掉y值缺失的数据
    :param df:
    :param y:
    :param output_column:
    :return:
    '''
    y = list(y)
    indices_to_drop = []
    indices_to_keep = []
    for idx, val in enumerate(y):
        if str(val) in bad_vals_as_strings:
            indices_to_drop.append(idx)

    if len(indices_to_drop) > 0:
        set_of_indices_to_drop = set(indices_to_drop)

        print ('We encountered a number of missing values for this output column')
        if output_column is not None:
            print (output_column)
        print ('And here is the number of missing (nan, None, etc.) values for this column: ' )
        print (len(indices_to_drop))
        print ('Here are some example missing values')
        for idx, df_idx in enumerate(indices_to_drop):
            if idx >= 5:
                break
            print(y[df_idx])
        print ('We will remove these values, and continue with training on the cleaned dataset')

        support_mask = [True if idx not in set_of_indices_to_drop else False for idx in range(df.shape[0])]
        if isinstance(df, pd.DataFrame):
            df = df[support_mask]
        elif scipy.sparse.issparse(df):
            df = delete_rows_csr(df, indices_to_drop)
        y = [val for idx, val in enumerate(y) if idx not in set_of_indices_to_drop]

    return df, y


class ExtendedLabelEncoder(LabelEncoder):
    def __init__(self):
        super(self.__class__, self).__init__()

    def transform(self, y):
        '''
        将标签值y进行encoding
        :param y:
        :return:
        '''
        y = column_or_1d(y, warn=True)

        classes = np.unique(y)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            self.classes_ = np.hstack((self.classes_, diff))
        return np.searchsorted(self.classes_, y)


class ExtendedPipeline(Pipeline):
    def __init__(self, steps, keep_cat_features=False):
        super(self.__class__, self).__init__(steps)
        self.keep_cat_features = keep_cat_features

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_uncertainty(self, X):
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_uncertainty(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def score_uncertainty(self, X):
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transfrom(Xt)
        return self.steps[-1][-1].score_uncertainty(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def transform_only(self, X):
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].transform_only(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_intervals(self, X, return_type=None):
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_intervals(Xt, return_type=return_type)


def clean_params(params):
    cleaned_params = {}
    for k, v in params.items():
        if k[:7] == 'model__':
            cleaned_params[k[7:]] = v
    return cleaned_params