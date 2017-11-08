__author__ = 'Stephen Lee (mingyangli1314@outlook.com)'

import datetime
import dateutil

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

# The easiest way to check against a bunch of different bad values is to convert whatever val we have into a string,
# then check it against a set containing the string representation of a bunch of bad values
bad_vals_as_strings = set([str(float('nan')), str(float('inf')), str(float('-inf')), 'None', 'none', 'NaN', 'nan', 'NULL', 'null', '', 'inf', '-inf'])

# 1. clean_val will try to turn a value into a float.
# 2. If it fails, it will attempt to strip commas and then attempt to turn it into a float again
# 3. Additionally, it will check to make sure the value is not in a set of bad vals (nan, None, inf, etc.)
# 4. This function will either return a clean value, or raise an error if we cannot turn the value into a float or the value is a bad val
def clean_val(val):
    if str(val) in bad_vals_as_strings:
        raise (ValueError('clean_val failed'))
    else:
        try:
            float_val = float(val)
        except ValueError:
            # This will throw a ValueError if it fails
            # Remove any commas in the string, and try to turn into a float again
            try:
                cleaned_string = val.replace(',', '')
                float_val = float(cleaned_string)
            except TypeError:
                return None
        return float_val


# 1. Same as above, except this version returns float('nan') when it fails
# 2. This plays more nicely with df.apply, and assumes we will be handling nans appropriately when doing DataFrameVectorizer later
def clean_val_nan_version(key, val):
    try:
        str_val = str(val)
    except UnicodeEncodeError as e:
        str_val  = val.encode('ascii', 'ignore').decode('ascii')
        print ('Here is the value that causes the UnicodeEncodeError to be thrown: ')
        print (val)
        print ('Here is the feature name: ')
        print (key)
        raise (e)

    if str_val in bad_vals_as_strings:
        return float('nan')
    else:
        try:
            float_val = float(val)
        except ValueError:
            # remove any commas in the string, and try to turn into a float again
            try:
                cleaned_string = val.replace(',', '')
            except TypeError:
                print('*' * 30)
                print('We excepted this value to be numeric, but were unable to convert it to a float:')
                print(val)
                print('Here is the feature name:')
                print(key)
                print('*' * 30)
                return float('nan')

            try:
                float_val = float(cleaned_string)
            except:
                return float('nan')

        except TypeError:
            # This is what happens if you feed in a datetime object to float
            print('*' * 30)
            print('We expected this value to be numeric, but were unable to convert it to a float:')
            print(val)
            print('Here is the feature name:')
            print(key)
            print('*' * 30)
            return float('nan')

        return float_val


class BasicDataCleaning(BaseEstimator, TransformerMixin):
    def __init__(self, column_descriptions=None):
        self.column_descriptions = column_descriptions
        self.transformed_column_descriptions = column_descriptions.copy()
        self.text_col_indicator = set(['text', 'nlp'])

        self.text_columns = {}

        for key, val in self.column_descriptions.items():
            if val in self.text_col_indicator:
                self.text_columns[key] = TfidfVectorizer(
                    decode_error='ignore',          # If we have any documents that can't be decoded properly, just ignore them and keep going as planned with everything else.
                    strip_accents='unicode',        # Try to strip accents from characters. Using unicode is slightly slower but more comprehensive than 'ascii'
                    analyzer='word',                 # Can also choose 'character', which will likely increase accuracy, at the cost of much more space, generally
                    stop_words='english',            # Remove commonly found english words ('it', 'a', 'the') which do not typically contain muchi signal
                    lowercase=True,                  # Convert all characters to lowercase
                    max_df=0.9,                       # Only consider words that appear in fewer than max_df percent of all documents
                                                      # In this case, ignore all words that appear in 90% of all documents
                    max_features=3000                 # Consider only the most frequently occurring 3000 words, after taking into account all the other filtering going on
            )

    def get(self, prop_name, default=None):
        try:
            return getattr(self, prop_name)
        except AttributeError:
            return default

    def fit(self, X_df, y=None):
        print ('Running basic data cleaning')

        # See if we should fit TfidfVectorizer or not
        for key in X_df.columns:
            if X_df[key].dtype == 'object' and self.column_descriptions.get(key, False) not in ['categorical', 'ignore', 'nlp']:
                # First, make sure that the values in this column are not just ints, or float('nan')
                vals = X_df[key].sample(n=10)
                is_categorical = False
                for val in vals:
                    try:
                        if val is not None:
                            float(val)
                    except Exception as e:
                        print (e)
                        is_categorical = True

                if is_categorical:
                    print ('\n')
                    print ('Encountered a column that is not marked as categorical, but is an "object" pandas type, which typically indicates a categorical column')
                    print ('The name of this columns is: "{}"'.format(key))
                    print ('Some example features in this column are: {}'.format(list(X_df[key].sample(n=5))))
                    print ('If this is a categorical column, please mark it as  `{}: "categorical"` as part of your column_descriptions'.format(key))
                    print ('If this is not a categorical column, please consider converting its dtype before passing data into smart_ml')
                    print ('\n')
                    warnings.warn('Consider marking the "{}" column as categorical'.format(key))

            if self.transformed_column_descriptions.get(key) is None:
                self.transformed_column_descriptions[key] = 'continuous'

            if key in self.text_columns:
                X_df[key].fillna('nan', inplace=True)
                # text_col = X_df[key].astype('unicode', errors='ignore')
                text_col = X_df[key]
                self.text_columns[key].fit(text_col)

                col_names = self.text_columns[key].get_feature_names()

                # Make weird characters play nice, or just ignore them
                for idx, word in enumerate(col_names):
                    try:
                        col_names[idx] = str(word)
                    except:
                        col_names[idx] = 'non_ascii_word_' + str(idx)

                col_names = ['nlp_' + key + '_' + str(word) for word in col_names]
                self.text_columns[key].cleaned_feature_names = col_names

        return self


    def transform(self, X, y=None):
        ignore_none_fields = False
        if self.get('Transformed_column_descriptions', None) is not None:
            ignore_none_fields = True
        column_descriptions = self.get('transformed_column_descriptions', self.column_descriptions)

        # Convert input to DataFrame if we were given a list of dictionaries
        if isinstance(X, list):
            X = pd.DataFrame(X)

        X = X.copy()

        # All of these are values we will not want to keep for training this particular estimator,
        # Note that we have already split out the output column and saved it into it's own variable
        vals_to_drop = set(['ignore', 'output', 'regressor', 'classifier'])

        # It is much more efficient to drop a bunch of columns at once, rather than one at a time
        cols_to_drop = []

        if isinstance(X, dict):
            dict_copy = {}
            for key, val in X.items():
                col_desc = column_descriptions.get(key, None)

                if col_desc is None and ignore_none_fields is True:
                    continue
                elif col_desc in (None, 'continuous', 'numetical', 'float', 'int'):
                    dict_copy[key] = clean_val_nan_version(key, val)
                elif col_desc == 'date':
                    date_feature_dict = add_date_features_dict(X, key)
                    dict_copy.update(date_feature_dict)
                elif col_desc == 'categorical':
                    dict_copy[key] = val
                elif key in self.text_columns:
                    col_names = self.text_columns[key].cleaned_feature_names
                    try:
                        text_val = str(X[key])
                    except UnicodeEncodeError:
                        text_val = X[key].encode('ascii', 'ignore').decode('ascii')

                    # the transform function expects a list
                    text_val = [text_val]

                    nlp_matrix = self.text_columns[key].transform(text_val)

                    # From here, it's all about transforming the output from the tf-idf transform into a dictionary
                    # It outputs a sparse csr matrics
                    # First, we transform to coo
                    nlp_matrix = nlp_matrix.tocoo()
                    # Then, we grab the relevant column names
                    relevant_col_names = []
                    for col_idx in nlp_matrix.col:
                        relevant_col_names.append(col_names[col_idx])

                    # Then we zip together the relevant columns and the sparse data into a dictionary
                    relevant_nlp_cols = {k:v for k, v in zip(relevant_col_names, nlp_matrix.data)}
                    dict_copy.update(relevant_nlp_cols)

                elif col_desc in vals_to_drop:
                    pass
            return dict_copy

        else:
            X.reset_index(drop=True, inplace=True)
            for key in X.columns:
                col_desc = column_descriptions.get(key)

                if col_desc is None and ignore_none_fields is True:
                    continue
                elif col_desc == 'categorical':
                    # We will handle categorical data later, one-hot-encoding it inside DataFrameVectorizer
                    pass
                elif col_desc in (None, 'continuous', 'numerical', 'float', 'int'):
                    # For all of our numerical columns, try to turn all of these values into floats
                    # This function handles commas inside strings that represent numbers, and returns nan if cannot turn this value into a float nans are ignored in DataFrameVectorizer
                    try:
                        X[key] = X[key].apply(lambda x: clean_val_nan_version(key, x))
                    except TypeError as e:
                        raise (e)
                    except UnicodeEncodeError as e:
                        print ('!' * 50)
                        print ('!' * 50)
                        print ('We have found a column that is not marked as a categorical column that has unicode values in it.')
                        print ('Here is the column name: ')
                        print (key)
                        print ('The actual value that caused the issue is logged right above the exclamation points.')
                        print ('Please either mark this column as categorical, or clean up the values in this column.')
                        print ('!' * 50)
                        print ('!' * 50)

                elif col_desc == 'date':
                    X = add_date_features_df(X, key)
                elif key in self.text_columns:
                    col_names = self.text_columns[key].cleaned_feature_names

                    # Make weird characters play nice, or just ignore them
                    for idx, word in enumerate(col_names):
                        try:
                            col_names[idx] = str(word)
                        except:
                            col_names[idx] = 'non_ascii_word_' + str(idx)
                    col_names = ['nlp_' + 'key' + '_' + str(word) for word in col_names]

                    X[key].fillna('nan', inplace=True)
                    nlp_matrix = self.text_columns[key].transform(X[key])
                    nlp_matrix = nlp_matrix.toarray()

                    text_df = pd.DataFrame(nlp_matrix)
                    text_df.columns = col_names

                    X = X.join(text_df)

                    # Once the transformed dataframe is added, remove the original text

                    X = X.drop(key, axis=1)

                elif col_desc in vals_to_drop:
                    cols_to_drop.append(key)
                else:
                    # If we have gotten here, the value is not any that we recognize
                    # This is most likely a typo that the user would want to be informed of , or a case while we're developing on smart_ml itself.
                    # In either case, it's useful to log it
                    print ('When transforming the data, we have encountered a value in column_descriptions that is not currently supported. '
                           'The column has been dropped to allow the rest of the pipeline to run. Here\'s the name of the column: ')
                    print (key)
                    print ('And here is the value for this column passed into columns_descriptions: ')
                    print (col_desc)
                    warnings.warn('UnknownValueInColumnDescriptions: Please make sure all the values you pass into column_descriptions are valid.')

        # Historically we've deleted columns here. However, we're moving this to DataFrameVectorizer as part of a broader effort to reduce duplicate computation
        return X

def minutes_into_day_parts(minutes_into_day):
    if minutes_into_day < 6 * 60:
        return 'late_night'
    elif minutes_into_day < 10 * 60:
        return 'morning'
    elif minutes_into_day < 11.5 * 60:
        return 'mid_morning'
    elif minutes_into_day < 14 * 60:
        return 'lunchtime'
    elif minutes_into_day < 18 * 60:
        return 'afternoon'
    elif minutes_into_day < 20.5 * 60:
        return 'dinnertime'
    elif minutes_into_day < 23.5 * 60:
        return 'early_night'
    else:
        return 'late_night'


# Note: assumes that the column is already formatted as a pandas data type
def add_date_features_df(df, date_col):
    # Pandas nicely tries to prevent you from doing stupid things, like setting values on a copy of a df, noot your real one
    # However, it's a bit overzealous in this case. so we'll side-step a bunch of warnings by setting is_copy to false here.
    df.is_copy = False

    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col + '_day_of_week'] = df[date_col].apply(lambda x: x.weekday()).astype(int, errors='ignore')

    try:
        df[date_col + '_hour'] = df[date_col].apply(lambda x: x.hour).astype(int, errors='ignore')
        df[date_col + '_minutes_into_day'] = df[date_col].apply(lambda x: x.hour * 60 + x.minutes)
    except AttributeError:
        pass

    df[date_col + '_is_weekend'] = df[date_col].apply(lambda x: x.weekday() in (5,6))
    df[date_col + '_day_part'] = df[date_col + '_minutes_into_day'].apply(minutes_into_day_parts)

    df = df.drop([date_col], axis=1)
    return df


# Same logic as above, except implemented for a single dictionary, which is much faster at prediction time when getting just a single prediction
def add_date_features_dict(row, date_col):
    date_feature_dict = {}

    # Handle cases where the val for the date_col is None
    try:
        date_val = row[date_col]
        if date_val == None:
            return date_feature_dict
        if not isinstance(date_val, (datetime.datetime, datetime.date)):
            date_val = dateutil.parser.parse(date_val)
    except:
        return date_feature_dict

    # Make a copy of all the engineered features from the date, without modifying the original object at all
    # This way the same original object can be passed into a number of different trained smart_ml predictors
    date_feature_dict[date_col + '_day_of_week'] = date_val.weekday()

    # nesting this inside a try/except block because the date might be a datetime.date, not a datetime.datetime
    try:
        date_feature_dict[date_col + '_hour'] = date_val.hour
        date_feature_dict[date_col + '_minutes_into_day'] = date_val.hour * 60 + date_val.minute
    except AttributeError:
        pass

    date_feature_dict[date_col + '_is_weekend'] = date_val.weekday() in (5, 6)
    return date_feature_dict


