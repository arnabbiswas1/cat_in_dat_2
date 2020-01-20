import gc
import random
import sys 
import feather
import os
import logging
import time

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import category_encoders as ce

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost


# Directory consisting of (almost) original data in feather format
CREATED_DATA_DIR = '/home/jupyter/kaggle/energy/data/read_only_feather/v2'
# Directory consisting of differnt newly created data
CREATED_FEATURE_DIR = '/home/jupyter/kaggle/energy/data/created_data'


def read_files(logger, dir_path, train_file_name='train.csv', 
               test_file_name='test.csv', 
               submission_file_name='sample_submission.csv', index_col=None):
    """
    Returns 3 data frames consisting of train, test and sample_submission.csv
    """
    logger.info('Loading Data...')
    train = pd.read_csv(f'{dir_path}/{train_file_name}', index_col=index_col)
    test = pd.read_csv(f'{dir_path}/{test_file_name}', index_col=index_col)
    submission = pd.read_csv(f'{dir_path}/{submission_file_name}')
    
    logger.info(f'Shape of {train_file_name} : {train.shape}')
    logger.info(f'Shape of {test_file_name} : {test.shape}')
    logger.info(f'Shape of {submission_file_name} : {submission.shape}')
    
    logger.info('Data Loaded...')
    
    return train, test, submission


def read_data(data_dir, train=True, test=True, weather_train=False, weather_test=False, building=False):
    """
    Don't use
    
    Designed for ASHRAE
    """
    print('Reading Data...')
    train_df = None
    test_df = None
    weather_train_df = None
    weather_test_df = None
    building_df = None
    if train:
        train_df = feather.read_dataframe(f'{data_dir}/train_merged.feather')
        print(f'Shape of train_df : {train_df.shape}')
    if test:
        test_df = feather.read_dataframe(f'{data_dir}/test_merged.feather')
        print(f'Shape of test_df : {test_df.shape}')
    if weather_train:
        weather_train_df = feather.read_dataframe(f'{data_dir}/weather_train.feather')
        print(f'Shape of weather_train_df : {weather_train_df.shape}')
    if weather_test:
        weather_test_df = feather.read_dataframe(f'{data_dir}/weather_test.feather')
        print(f'Shape of weather_test_df : {weather_test_df.shape}')
    if building:
        building_df = feather.read_dataframe(f'{data_dir}/building.feather')
        print(f'Shape of building_df : {building_df.shape}')
    return train_df, test_df, weather_train_df, weather_test_df, building_df


############################################## Utility ##############################################


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    

def trigger_gc():
    """
    Trigger GC
    """
    print(gc.collect())
    
    
def set_timezone():
    """
    Sets the time zone to Kolkata.
    """
    os.environ["TZ"] = "Asia/Calcutta"
    time.tzset()


def get_logger(logger_name, model_number, run_id, path):
    """
    https://realpython.com/python-logging/
    https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    """
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    s_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(f'{path}/{model_number}_{run_id}.log')
    formatter = logging.Formatter(FORMAT)
    s_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    logger.addHandler(s_handler)
    return logger


def update_tracking(run_id,
                    field,
                    value, csv_file="../../tracking/tracking.csv", 
                    is_integer=False, no_of_digits=None, 
                    drop_incomplete_rows=False):
    """
    Function to update the tracking CSV with information about the model
    
    https://github.com/RobMulla/kaggle-ieee-fraud-detection/blob/master/scripts/M001.py#L98
    
    """
    try:
        df = pd.read_csv(csv_file, index_col=[0])
        
        # If the file exists, drop rows (without final results) 
        # for previous runs which has been stopped inbetween. 
        if (drop_incomplete_rows & ('oof score' in df.columns)):
            df = df.loc[~df['oof score'].isna()]
             
    except FileNotFoundError:
        df = pd.DataFrame()
        
    if is_integer:
        value = round(value)
    elif no_of_digits is not None:
        value = round(value, no_of_digits)

    # Model number is index
    df.loc[run_id, field] = value  
    df.to_csv(csv_file)
    
    
def save_file(logger, df, dir_name, file_name):
    """
    Utility method to save submission, off files etc.
    """
    logger.info(f'Saving {dir_name}/{file_name}')
    df.to_csv(f'{dir_name}/{file_name}', index=False)
    
    
def save_artifacts(logger, is_test, is_plot_fi, 
                   result_dict, 
                   submission_df, 
                   model_number, 
                   run_id, sub_dir, oof_dir, fi_dir, fi_fig_dir):
    """
    Save the submission, OOF predictions, feature importance values
    and plos to different directories.
    """
    score = result_dict['avg_cv_scores']
    
    if is_test == False:
        # Save submission file
        submission_df.target = result_dict['prediction']
        save_file(logger, 
                  submission_df,
                  sub_dir, 
                  f'sub_{model_number}_{run_id}_{score:.4f}.csv')

        # Save OOF
        oof_df = pd.DataFrame(result_dict['yoof'])
        save_file(logger, 
                  oof_df, 
                  oof_dir, 
                  f'oof_{model_number}_{run_id}_{score:.4f}.csv')
    
    if is_plot_fi == True:
        # Feature Importance
        feature_importance_df = result_dict['feature_importance']
        save_file(logger, 
                  feature_importance_df,
                  fi_dir, 
                  f'fi_{model_number}_{run_id}_{score:.4f}.csv')

        # Save the plot
        best_features = result_dict['best_features']
        save_feature_importance_as_fig(best_features, 
                                       fi_fig_dir, 
                                       f'fi_{model_number}_{run_id}_{score:.4f}.png')
    
    
############################################## Visualization ##############################################


def plot_meter_reading_for_site(df, site_id, meter_name):
    """
    Plot meter_reading for an entire site for all buildings
    """
    df = df.set_index('timestamp')
    building_id_list = df.columns
    for building_id in building_id_list:
        fig = go.Figure()
        df_subset = df.loc[:, building_id]
        fig.add_trace(go.Scatter(
             x=df_subset.index,
             y=df_subset.values,
             name=f"{meter_name}",
             hoverinfo=f'x+y+name',
             opacity=0.7))

        fig.update_layout(width=1000,
                        height=500,
                        title_text=f"Meter Reading for Site [{site_id}] Building [{building_id}]",
                        xaxis_title="timestamp",
                        yaxis_title="meter_reading",)
        fig.show()
        

def plot_meter_reading_for_building(df, site_id, building_id, meter_name):
    """
    Plot meter_reading for an entire site for all buildings
    """
    df = df.set_index('timestamp')
    building_id_list = df.columns
    for building_id in building_id_list:
        fig = go.Figure()
        df_subset = df.loc[:, building_id]
        fig.add_trace(go.Scatter(
             x=df_subset.index,
             y=df_subset.values,
             name=f"{meter_name}",
             hoverinfo=f'x+y+name',
             opacity=0.7))

        fig.update_layout(width=1000,
                        height=500,
                        title_text=f"Meter Reading for Site [{site_id}] Building [{building_id}]",
                        xaxis_title="timestamp",
                        yaxis_title="meter_reading",)
        fig.show()
        

def display_all_site_meter_reading(df, site_id=0, meter=0):
    """
    Plot meter reading for the entire site for a particular type of meter 
    """
    df_meter_subset = df[(df.site_id == site_id) & (df.meter == meter)]
    df_meter_subset = df_meter_subset.pivot(index='timestamp', columns='building_id', values='meter_reading')

    column_names = df_meter_subset.reset_index().columns.values
    df_meter_subset.reset_index(inplace=True)
    df_meter_subset.columns = column_names
    
    print(f'Missing Values for {site_id}')
    display(df_meter_subset.isna().sum())
    
    plot_meter_reading_for_site(df_meter_subset, site_id, meter_dict[meter])


def plot_hist_train_test_overlapping(df_train, df_test, feature_name, kind='hist'):
    """
    Plot histogram for a particular feature both for train and test.
    
    kind : Type of the plot
    
    """
    df_train[feature_name].plot(kind=kind, figsize=(15, 5), label='train', 
                         bins=50, alpha=0.4, 
                         title=f'Train vs Test {feature_name} distribution')
    df_test[feature_name].plot(kind='hist',label='test', bins=50, alpha=0.4)
    plt.legend()
    plt.show()
    

def plot_barh_train_test_side_by_side(df_train, df_test, feature_name, normalize=True, sort_index=False):
    """
    Plot histogram for a particular feature both for train and test.
    
    kind : Type of the plot
    
    """
    print(f'Number of unique values in train : {count_unique_values(df_train, feature_name)}')
    print(f'Number of unique values in test : {count_unique_values(df_test, feature_name)}')
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 8))
    
    if sort_index == True:
            df_train[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_index().plot(
                kind='barh', figsize=(15, 5), 
                ax=ax1,
                grid=True,
                title=f'Bar plot for {feature_name} for train')
    
            df_test[feature_name].value_counts(
                    normalize=normalize, dropna=False).sort_index().plot(
                    kind='barh', figsize=(15, 5), 
                    ax=ax2,
                    grid=True,
                    title=f'Bar plot for {feature_name} for test')
    else:
        df_train[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_values().plot(
                kind='barh', figsize=(15, 5), 
                ax=ax1,
                grid=True,
                title=f'Bar plot for {feature_name} for train')

        df_test[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_values().plot(
                kind='barh', figsize=(15, 5), 
                ax=ax2,
                grid=True,
                title=f'Bar plot for {feature_name} for test')

    
    plt.legend()
    plt.show()
    
    
def plot_line_train_test_overlapping(df_train, df_test, feature_name):
    """
    Plot line for a particular feature both for train and test
    """
    df_train[feature_name].plot(kind='line', figsize=(10, 5), label='train', 
                          alpha=0.4, 
                         title=f'Train vs Test {feature_name} distribution')
    df_test[feature_name].plot(kind='line',label='test', alpha=0.4)
    plt.ylabel(f'Value of {feature_name}')
    plt.legend()
    plt.show()
    
    
def plot_hist(df, feature_name, kind='hist', bins=100, log=True):
    """
    Plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(kind='hist', 
                                              bins=bins, 
                                              figsize=(15, 5), 
                                              title=f'Distribution of log1p[{feature_name}]')
    else:
        df[feature_name].plot(kind='hist', 
                              bins=bins, 
                              figsize=(15, 5), 
                              title=f'Distribution of {feature_name}')
    plt.show()


def plot_barh(df, feature_name, normalize=True, kind='barh', figsize=(15,5), sort_index=False):
    """
    Plot barh for a particular feature both for train and test.
    
    kind : Type of the plot
    
    """
    if sort_index==True:
        df[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_index().plot(
                kind=kind, figsize=figsize, grid=True,
                title=f'Bar plot for {feature_name}')
    else:   
        df[feature_name].value_counts(
                normalize=normalize, dropna=False).sort_values().plot(
                kind=kind, figsize=figsize, grid=True,
                title=f'Bar plot for {feature_name}')
    
    plt.legend()
    plt.show()
    

def plot_boxh(df, feature_name, kind='box', log=True):
    """
    Box plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(kind='box', vert=False, 
                                                  figsize=(10, 6), 
                                                  title=f'Distribution of log1p[{feature_name}]')
    else:
        df[feature_name].plot(kind='box', vert=False, 
                              figsize=(10, 6), 
                              title=f'Distribution of {feature_name}')
    plt.show()
    
    
def plot_boxh_groupby(df, feature_name, by):
    """
    Box plot with groupby feature
    """
    df.boxplot(column=feature_name, by=by, vert=False, 
                              figsize=(10, 6))
    plt.title(f'Distribution of {feature_name} by {by}')
    plt.show()
    
    
def save_feature_importance_as_fig(best_features_df, dir_name, file_name):
    plt.figure(figsize=(16, 12));
    sns.barplot(x="importance", y="feature", data=best_features_df.sort_values(by="importance", ascending=False));
    plt.title('Features (avg over folds)');
    plt.savefig(f'{dir_name}/{file_name}')

    
########################################################### EDA ###########################################################

def display_head(df):
    display(df.head(2))

    
def check_null(df):
    print('Checking Null Percentage..')
    return df.isna().sum() * 100/len(df)


def check_duplicate(df, subset):
    print(f'Number of duplicate rows considering {len(subset)} features..')
    if subset is not None: 
        return df.duplicated(subset=subset, keep=False).sum()
    else:
        return df.duplicated(keep=False).sum()

    
def count_unique_values(df, feature_name):
    return df[feature_name].nunique()


def do_value_counts(df, feature_name):
    return df[feature_name].value_counts(normalize=True, dropna=False).sort_values(ascending=False) * 100


def check_id(df, column_name, data_set_name):
    '''
    Check if the identifier column is continous and monotonically increasing
    '''
    print(f'Is the {column_name} monotonic : {df[column_name].is_monotonic}')
    # Plot the column
    ax = df[column_name].plot(title=data_set_name)
    plt.show()
    
    
def get_fetaure_names(df, feature_name_substring) :
    """
    Returns the list of features with name matching 'feature_name_substring'
    """
    return [col_name for col_name in df.columns if col_name.find(feature_name_substring) != -1]


def check_value_counts_across_train_test(train_df, test_df, feature_name, normalize=True):
    """
    Create a DF consisting of value_counts of a particular feature for 
    train and test
    """
    train_counts = train_df[feature_name].sort_index().value_counts(normalize=normalize, dropna=True) * 100
    test_counts = test_df[feature_name].sort_index().value_counts(normalize=normalize, dropna=True) * 100
    count_df = pd.concat([train_counts, test_counts], axis=1).reset_index(drop=True)
    count_df.columns = [feature_name, 'train', 'test']
    return count_df


def get_non_zero_meter_reading_timestamp(df, building_id, start_time, stop_time, meter=0):
    """
    For ASHRAE
    
    For a particular building, when was the first non-zero meter reading appeared.
    given the start and stop time and the type of the meter
    """
    return df[(df.building_id == building_id) 
                & (df.timestamp >= np.datetime64(start_time)) 
                & (df.timestamp < np.datetime64(stop_time)) 
                & (df.meter_reading > 0) & (df.meter == meter)]['timestamp'].iloc[0]


###################################################### Pre-Processing ######################################################

def get_encoder(encoder_name):
    """
    Returns an Encdoer Object given the name of the encoder
    """
    if encoder_name == 'LabelEncoder':
        return LabelEncoder()
    elif(encoder_name == 'CountEncoder'):
        return ce.CountEncoder()
    else:
        return None


def do_encoding(encoder_name, source_train_df, source_test_df, 
                      target_train_df, target_test_df, 
                      categorical_features, feature_name_suffix):
        """
        Given with a type of encoding, encode set of features
        listed in categorical_features variable
        """
        for feature_name in categorical_features:
            encoder = get_encoder(encoder_name)
            encoder.fit(list(source_train_df[feature_name].values) + list(source_test_df[feature_name].values))
            if feature_name_suffix:
                target_feature_name = f'{feature_name}_{feature_name_suffix}'
                print(f'{encoder_name} of feature [{feature_name}] is saved at [{target_feature_name}]')
            else:
                target_feature_name = feature_name
                print(f'{encoder_name} the feature [{target_feature_name}]')
            target_train_df[target_feature_name] = encoder.transform(list(source_train_df[feature_name].values))
            target_test_df[target_feature_name] = encoder.transform(list(source_test_df[feature_name].values))
        return target_train_df, target_test_df


def do_label_encoding(source_train_df, source_test_df, 
                      target_train_df, target_test_df, 
                      categorical_features, feature_name_suffix=None):
    """
    Label encode the categorical features.
    After encdoing, it appends a new set of features with name 
    <original_feature_name>_label to the target dataframe
    """
    return do_encoding('LabelEncoder', source_train_df, source_test_df, 
                      target_train_df, target_test_df, 
                      categorical_features, feature_name_suffix)


def do_count_encoding(source_train_df, source_test_df, 
                      target_train_df, target_test_df, 
                      categorical_features, feature_name_suffix=None):
    """
    Count encode the categorical features.
    After encdoing, it appends a new set of features with name 
    <original_feature_name>_label to the target dataframe
    """
    return do_encoding('CountEncoder', source_train_df, source_test_df, 
                      target_train_df, target_test_df, 
                      categorical_features, feature_name_suffix)


def convert_to_int(df, feature_names):
    for feature_name in feature_names:
        df.loc[:, feature_name] = df[feature_name].astype('int')
    return df


################################################## Time Series ####################################################

def create_date_features(source_df, target_df, feature_name):
    '''
    Create new features related to dates
    
    source_df : DataFrame consisting of the timestamp related feature
    target_df : DataFrame where new features will be added
    feature_name : Name of the feature of date type which needs to be decomposed.
    '''
    target_df.loc[:, 'year'] = source_df.loc[:, feature_name].dt.year.astype('uint16')
    target_df.loc[:, 'month'] = source_df.loc[:, feature_name].dt.month.astype('uint8')
    target_df.loc[:, 'quarter'] = source_df.loc[:, feature_name].dt.quarter.astype('uint8')
    target_df.loc[:, 'weekofyear'] = source_df.loc[:, feature_name].dt.weekofyear.astype('uint8')
    
    target_df.loc[:, 'hour'] = source_df.loc[:, feature_name].dt.hour.astype('uint8')
    #target_df.loc[:, 'minute'] = source_df.loc[:, feature_name].dt.minute.astype('uint32')
    #target_df.loc[:, 'second'] = source_df.loc[:, feature_name].dt.second.astype('uint32')
    
    target_df.loc[:, 'day'] = source_df.loc[:, feature_name].dt.day.astype('uint8')
    target_df.loc[:, 'dayofweek'] = source_df.loc[:, feature_name].dt.dayofweek.astype('uint8')
    target_df.loc[:, 'dayofyear'] = source_df.loc[:, feature_name].dt.dayofyear.astype('uint8')
    target_df.loc[:, 'is_month_start'] = source_df.loc[:, feature_name].dt.is_month_start
    target_df.loc[:, 'is_month_end'] = source_df.loc[:, feature_name].dt.is_month_end
    target_df.loc[:, 'is_quarter_start']= source_df.loc[:, feature_name].dt.is_quarter_start
    target_df.loc[:, 'is_quarter_end'] = source_df.loc[:, feature_name].dt.is_quarter_end
    target_df.loc[:, 'is_year_start'] = source_df.loc[:, feature_name].dt.is_year_start
    target_df.loc[:, 'is_year_end'] = source_df.loc[:, feature_name].dt.is_year_end
    
    # This is of type object
    #target_df.loc[:, 'month_year'] = source_df.loc[:, feature_name].dt.to_period('M')
    
    return target_df


def fill_with_gauss(df, w=12):
    """
    Fill missing values in a time series data using gaussian
    """
    return df.fillna(df.rolling(window=w, win_type='gaussian', center=True, min_periods=1).mean(std=2))


def fill_with_po3(df):
    """
    Fill missing values in a time series data using interpolation (polynomial, order 3)
    """
    df = df.fillna(df.interpolate(method='polynomial', order=3))
    assert df.count().min() >= len(df) - 1 
    # fill the first item with second item
    return df.fillna(df.iloc[1])         


def fill_with_lin(df):
    """
    Fill missing values in a time series data using interpolation (linear)
    """
    df =  df.fillna(df.interpolate(method='linear'))
    assert df.count().min() >= len(df) - 1 
    # fill the first item with second item
    return df.fillna(df.iloc[1])         


def fill_with_mix(df):
    """
    Fill missing values in a time series data using interpolation (linear + polynomial)
    """
    df = (df.fillna(df.interpolate(method='linear', limit_direction='both')) +
               df.fillna(df.interpolate(method='polynomial', order=3, limit_direction='both'))
              ) * 0.5
    assert df.count().min() >= len(df) - 1 
    # fill the first item with second item
    return df.fillna(df.iloc[1])


############################################ Feature Engineering ############################################


def concat_features(source_df, target_df, f1, f2):
    print(f'Concating features {f1} and {f2}')
    target_df[f'{f1}_{f2}'] =  source_df[f1].astype(str) + '_' + source_df[f2].astype(str)
    return target_df


def create_interaction_features(source_df, target_df):
    """
    For ASHARE
    """
    print('Creating interaction features...')
    target_df = concat_features(source_df, target_df, 'site_id', 'building_id')
    target_df['site_building_meter_id'] = source_df.site_id.astype(str) + '_' + source_df.building_id.astype(str) + '_' + source_df.meter.astype(str)
    target_df['site_building_meter_id_usage'] = source_df.site_id.astype(str) + '_' + source_df.building_id.astype(str) + '_' + source_df.meter.astype(str) + '_' + source_df.primary_use

    target_df = concat_features(source_df, target_df, 'site_id', 'meter')
    target_df = concat_features(source_df, target_df, 'building_id', 'meter')
    
    target_df = concat_features(source_df, target_df, 'site_id', 'primary_use')
    target_df = concat_features(source_df, target_df, 'building_id', 'primary_use')
    target_df = concat_features(source_df, target_df, 'meter', 'primary_use')
    
    return target_df


def create_age(source_df, target_df, f):
    """
    For ASHARE
    """
    print('Creating age feature')
    target_df['building_age'] = 2019 - source_df[f]
    return target_df


############################################ Modeling ############################################

def get_data_splits_by_fraction(dataframe, valid_fraction=0.1):
    """
    Creating holdout set from the train data based on fraction
    """
    print(f'Splitting the data into train and holdout with validation fraction {valid_fraction}...')
    valid_size = int(len(dataframe) * valid_fraction)
    train = dataframe[:valid_size]
    validation = dataframe[valid_size:]
    print(f'Shape of the training data {train.shape} ')
    print(f'Shape of the validation data {validation.shape}')
    return train, validation


def get_data_splits_by_month(dataframe, train_months, validation_months):
    """
    Creating holdout set from the train data based on months
    """
    print(f'Splitting the data into train and holdout based on months...')
    print(f'Training months {train_months}')
    print(f'Validation months {validation_months}')
    training = dataframe[dataframe.month.isin(train_months)]
    validation = dataframe[dataframe.month.isin(validation_months)]
    print(f'Shape of the training data {training.shape} ')
    print(f'Shape of the validation data {validation.shape}')
    return training, validation


def train_model(training, validation,predictors, target,  params, test_X=None):
    
    train_X = training[predictors]
    train_Y = np.log1p(training[target])
    validation_X = validation[predictors]
    validation_Y = np.log1p(validation[target])

    print(f'Shape of train_X : {train_X.shape}')
    print(f'Shape of train_Y : {train_Y.shape}')
    print(f'Shape of validation_X : {validation_X.shape}')
    print(f'Shape of validation_Y : {validation_Y.shape}')
    
    dtrain = lgb.Dataset(train_X, label=train_Y)
    dvalid = lgb.Dataset(validation_X, validation_Y)
    
    print("Training model!")
    bst = lgb.train(params, dtrain, valid_sets=[dvalid], verbose_eval=100)
    
    valid_prediction = bst.predict(validation_X)
    valid_score = np.sqrt(metrics.mean_squared_error(validation_Y, valid_prediction))
    print(f'Validation Score {valid_score}')
    
    if test_X is not None:
        print('Do Nothing')
    else:
        return bst, valid_score


def make_prediction_classification(logger, run_id, df_train_X, df_train_Y, df_test_X, kf, features, 
                                   params, n_estimators=10000, 
                                   early_stopping_rounds=100, model_type='lgb', 
                                   is_test=False, seed=42, model=None, 
                                   plot_feature_importance=False):
    """
    Make prediction for classification use case only
    
    model : Needed only for model_type=='sklearn'
    plot_feature_importance : Only needed for LGBM (in case feature importnace is needed)
    n_estimators : For XGB should be explicitly passed through this method
    early_stopping_rounds : For XGB should be explicitly passed through this method. 
                            For LGB can be passed through params as well
    
    """
    yoof = np.zeros(len(df_train_X))
    yhat = np.zeros(len(df_test_X))
    cv_scores = []
    result_dict = {}
    feature_importance = pd.DataFrame()
    best_iterations = []
    
    #kf = KFold(n_splits=n_splits, random_state=SEED, shuffle=False)
    #kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    fold = 0
    for in_index, oof_index in kf.split(df_train_X[features], df_train_Y):
        # Start a counter describing number of folds
        fold += 1
        # Number of splits defined as a part of KFold/StratifiedKFold
        n_splits = kf.get_n_splits()
        logger.info(f'fold {fold} of {n_splits}')
        X_in, X_oof = df_train_X.iloc[in_index].values, df_train_X.iloc[oof_index].values
        y_in, y_oof = df_train_Y.iloc[in_index].values, df_train_Y.iloc[oof_index].values
        
        if model_type == 'lgb':
            lgb_train = lgb.Dataset(X_in, y_in)
            lgb_eval = lgb.Dataset(X_oof, y_oof, reference=lgb_train)

            model = lgb.train(
                params,
                lgb_train,
                valid_sets = [lgb_train, lgb_eval],
                verbose_eval = 50,
                early_stopping_rounds=early_stopping_rounds
            )   
            
            del lgb_train, lgb_eval, in_index, X_in, y_in 
            gc.collect()
            
            yoof[oof_index] = model.predict(X_oof, num_iteration=model.best_iteration)
            if is_test==False:
                logger.info(f'Best number of iterations for fold {fold} is: {model.best_iteration}')
                yhat += model.predict(df_test_X.values, num_iteration=model.best_iteration)
                #TODO : Best iteration
                #yhat += model.predict(df_test_X.values)
            
            best_iteration = model.best_iteration
        
        elif model_type == 'xgb':
            xgb_train = xgb.DMatrix(data=X_in, label=y_in, feature_names=features)
            xgb_eval = xgb.DMatrix(data=X_oof, label=y_oof, feature_names=features)

            watchlist = [(xgb_train, 'train'), (xgb_eval, 'valid_data')]
            model = xgb.train(dtrain=xgb_train, 
                              num_boost_round=n_estimators,
                              evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds, 
                              params=params, 
                              verbose_eval=50)
            
            del xgb_train, xgb_eval, in_index, X_in, y_in 
            gc.collect()
            
            yoof[oof_index] = model.predict(xgb.DMatrix(X_oof, feature_names=features), ntree_limit=model.best_ntree_limit)
            if is_test==False:
                yhat += model.predict(xgb.DMatrix(
                    df_test_X.values, feature_names=features), 
                                      ntree_limit=model.best_ntree_limit)
            
            logger.info(f'Best number of iterations for fold {fold} is: {model.best_ntree_limit}')
            best_iteration = model.best_ntree_limit
            
        elif model_type == 'cat':
            model = CatBoost(params=params)
            #model.fit(X_in, y_in)
            model.fit(X_in, y_in, eval_set=(X_oof, y_oof), cat_features=[], use_best_model=True, verbose=False)
            
            del in_index, X_in, y_in 
            gc.collect()
            
            yoof[oof_index] = model.predict(X_oof)
            if is_test==False:
                yhat += model.predict(df_test_X.values)
            
            #best_iteration = model.best_iteration_
            
        elif model_type == 'sklearn':
            model = model
            model.fit(X_in, y_in)
            
            yoof[oof_index] = model.predict_proba(X_oof)[:, 1]
            if is_test==False:
                yhat += model.predict_proba(df_test_X.values)[:, 1]
            
        # Calculate feature importance per fold
        if model_type == 'lgb':
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = features
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        elif model_type == 'xgb':
            # Calculate feature importance per fold
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = model.get_score().keys()
            fold_importance["importance"] = model.get_score().values()
            fold_importance["fold"] = fold
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        
        cv_oof_score = roc_auc_score(y_oof, yoof[oof_index])
        logger.info(f'CV OOF Score for fold {fold} is {cv_oof_score}')
        cv_scores.append(cv_oof_score)
        best_iterations.append(best_iteration)
        
        del oof_index, X_oof, y_oof
        gc.collect()
        
        update_tracking(run_id, "metric_fold_{}".format(fold),
                    cv_oof_score,
                    is_integer=False)

    yhat /= n_splits

    oof_score = round(roc_auc_score(df_train_Y, yoof), 5)
    avg_cv_scores = round(sum(cv_scores)/len(cv_scores), 5)
    std_cv_scores = round(np.array(cv_scores).std(), 5)

    logger.info(f'Combined OOF score : {oof_score}')
    logger.info(f'Average of {fold} folds OOF score {avg_cv_scores}')
    logger.info(f'std of {fold} folds OOF score {std_cv_scores}')
    
    result_dict['yoof'] = yoof
    result_dict['prediction'] = yhat
    result_dict['oof_score'] = oof_score
    result_dict['cv_scores'] = cv_scores
    result_dict['avg_cv_scores'] = avg_cv_scores
    result_dict['std_cv_scores'] = std_cv_scores
    
    update_tracking(run_id, "oof_score", oof_score, is_integer=False)
    update_tracking(run_id, "cv_avg_score", avg_cv_scores, is_integer=False)
    update_tracking(run_id, "cv_std_score", std_cv_scores, is_integer=False)
    # Best Iteration
    update_tracking(run_id, 'avg_best_iteration', np.mean(best_iterations), is_integer=False)
    update_tracking(run_id, 'std_best_iteration', np.std(best_iterations), is_integer=False)
    
    del yoof, yhat
    gc.collect()
    
    # Plot feature importance
    if (model_type == 'lgb') | (model_type == 'xgb'):
        # Not sure why it was necessary. Hence commenting
        #feature_importance["importance"] /= n_splits
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:50].index

        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

        result_dict['feature_importance'] = feature_importance
        result_dict['best_features'] = best_features
            
    logger.info('Training/Prediction completed!')
    return result_dict
