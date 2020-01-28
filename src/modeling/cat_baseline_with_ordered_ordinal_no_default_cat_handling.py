import sys
import os
from datetime import datetime
import logging
from timeit import default_timer as timer

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

"""
- Missing Value Handling
    - Filled with missing_binary, missing_nom for binary and nominal
    - ord_0 : 999
    - ord_1, ord_2, ord_3, ord_4, ord_5 : missing_ord
    - day : 999
    - month : 999
- PreProcessing
    - ordinal variables
        - ord_1, ord_2 are ordered manually
        - 'ord_0', 'ord_3', 'ord_4', 'ord_5' : ordered based on string literal
    - Encoding
        - Convereted every variable to Cat type
        - Label Encoding
- Modeling
    - CatBoost
"""


sys.path.insert(0, "/home/jupyter/kaggle/cat_in_dat_2_git/cat_in_dat_2/src")
import utility

# First thing first. Set the time zone to City of Joy.
utility.set_timezone()

start = timer()

##################
# PARAMETERS
###################

EXP_DETAILS='Baseline with CatBoost manually ordered ord_1, ord_2. All the categorical features have been encoded using LabelEncoder'

run_id = "{:%m%d_%H%M}".format(datetime.now())
KERNEL_RUN = False
MODEL_NUMBER = os.path.basename(__file__).split('.')[0]


DATA_DIR = '/home/jupyter/kaggle/cat_in_dat_2_git/cat_in_dat_2/data/read_only'
SEED = 42

# Flags
IS_TEST=False
PLOT_FEATURE_IMPORTANCE = True

# General configuration stuff
LOGGER_NAME = 'modeling'
LOG_DIR = '../../log'
SUBMISSION_DIR = '../../sub'
OOF_DIR = '../../oof'
FI_DIR = '../../fi'
FI_FIG_DIR = '../../fi_fig'

# Parameters related to KFold
N_FOLDS = 10
SHUFFLE = True

# Parameters related to model
MODEL_TYPE = "cat"
METRIC = 'AUC'
N_ESTIMATORS = 100000
EARLY_STOPPING_ROUNDS = 100
VERBOSE = 100
N_THREADS = -1

# Name of the target
TARGET = 'target'

# Params 
cat_params = {
    'objective' : 'Logloss',
    'boosting_type' : 'Plain',
    'eval_metric' : METRIC,
    'thread_count': N_THREADS,
    'verbose' : VERBOSE,    # stdout about training process every 100 iter
    #'logging_level' : 'Verbose',
    'random_seed': SEED,
    'n_estimators' : N_ESTIMATORS,
    'early_stopping_rounds' : EARLY_STOPPING_ROUNDS
    }

logger = utility.get_logger(LOGGER_NAME, MODEL_NUMBER, run_id, LOG_DIR)

utility.set_seed(SEED)
logger.info(f'Running for Model Number {MODEL_NUMBER}')

utility.update_tracking(run_id, "model_number", MODEL_NUMBER, drop_incomplete_rows=True)
utility.update_tracking(run_id, "model_type", MODEL_TYPE)
utility.update_tracking(run_id, "model_type", MODEL_TYPE)
utility.update_tracking(run_id, "is_test", IS_TEST)
utility.update_tracking(run_id, "n_estimators", N_ESTIMATORS)
utility.update_tracking(run_id, "early_stopping_rounds", EARLY_STOPPING_ROUNDS)
utility.update_tracking(run_id, "random_state", SEED)
utility.update_tracking(run_id, "n_threads", N_THREADS)
#utility.update_tracking(run_id, "learning_rate", LEARNING_RATE)
utility.update_tracking(run_id, "n_fold", N_FOLDS)

############################################
# Preparaing Data
############################################

#Read the data file
train, test, submission = utility.read_files(logger=logger, dir_path=DATA_DIR, index_col='id')

combined_df = pd.concat([train.drop('target', axis=1), test])
logger.info(f'Shape of the combined DF {combined_df.shape}')

train_index = train.shape[0]
train_Y = train[TARGET]

# Fill the missing values
nom_features = utility.get_fetaure_names(train, 'nom')
logger.info(f'Number of nominal features {len(nom_features)}')
logger.info(f'Nominal Features : {nom_features}')

binary_features = utility.get_fetaure_names(train, 'bin')
logger.info(f'Number of binary features {len(binary_features)}')
logger.info(f'Binary Features : {binary_features}')

ordinal_fetaures = utility.get_fetaure_names(train, 'ord')
logger.info(f'Number of ordinal features {len(ordinal_fetaures)}')
logger.info(f'Ordinal Features : {ordinal_fetaures}')

#Filling missing values
combined_df[['bin_3', 'bin_4']] = combined_df[['bin_3', 'bin_4']].fillna('missing_binary')
combined_df[['bin_0', 'bin_1', 'bin_2']] = combined_df[['bin_0', 'bin_1', 'bin_2']].fillna(-1)

# Filling nominal variables with missing values
combined_df[nom_features] = combined_df[nom_features].fillna('missing_nom')

# ord_0 has apparently value fo type integer. 
combined_df['ord_0'] = combined_df['ord_0'].fillna(999)

# Fill missing values for other ordinal values
combined_df[['ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']] = combined_df[['ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']].fillna('missing_ord')

combined_df['day'] = combined_df['day'].fillna(999) 
combined_df['month'] = combined_df['month'].fillna(999)

# List to maintain names
new_features = []
features_to_removed = []

# For  ord_1, ord_2 we can decide on the order based on names
cat_type_ord_1 = pd.CategoricalDtype(categories=['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster', 'missing_ord'])
combined_df['ord_1_cat'] = combined_df['ord_1'].astype(cat_type_ord_1)

cat_type_ord_2 = pd.CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot', 'missing_ord'])
combined_df['ord_2_cat'] = combined_df['ord_2'].astype(cat_type_ord_2)

new_features = new_features + ['ord_1_cat', 'ord_2_cat']
features_to_removed = features_to_removed + ['ord_1', 'ord_2']

# Convert rest of the ordinal features in categories 
for feature_name in ['ord_0', 'ord_3', 'ord_4', 'ord_5']:
    logger.info(f'Converting {feature_name} in ordered categorical')
    combined_df[feature_name + '_cat'] = pd.Categorical(combined_df[feature_name], ordered=True)
    new_features = new_features + [feature_name + '_cat']
    features_to_removed = features_to_removed + [feature_name]

# Print the order of the ordinal features
for name in utility.get_fetaure_names(combined_df, '_cat'):
    logger.info(f'Categories for feature {name} : {combined_df[name].cat.categories}')

logger.info(f'List of new_features : {new_features}')
logger.info(f'List of features_to_removed : {features_to_removed}')

feature_list = [name for name in combined_df.select_dtypes(['object', 'float64']) if name not in features_to_removed]
# Print rest of the variables into categorical
for feature_name in feature_list:
    logger.info(f'Converting {feature_name} in categorical')
    combined_df[feature_name + '_cat'] = pd.Categorical(combined_df[feature_name])
    new_features = new_features + [feature_name + '_cat']
    features_to_removed = features_to_removed + [feature_name]

# Keep a copy of the original DF
combined_df_org = combined_df.copy(deep=True)

# remove the features not needed
combined_df = combined_df.drop(features_to_removed, axis=1)

for name in combined_df.columns:
    lb = LabelEncoder()
    combined_df[name] = lb.fit_transform(combined_df[name])

train_X = combined_df[:train_index]
test_X = combined_df[train_index:]

logger.info(f"train_X : {train_X.shape}")
logger.info(f"test_X : {test_X.shape}")
logger.info(f"train_Y : {train_Y.shape}")


#####################
# Build models
#####################

# Params are defines as dictionary above
kf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=SHUFFLE)
features = train_X.columns

#logger.info('################ Running with features ################')
logger.info(f'Feature names {features.values}')
logger.info(f'Target is {TARGET}')

utility.update_tracking(run_id, "no_of_features", len(features), is_integer=True)

result_dict = utility.make_prediction_classification(logger, run_id, train_X, train_Y, test_X, features=features,
                                                     params=cat_params, seed=SEED, 
                                                     kf=kf, model_type=MODEL_TYPE, 
                                                     plot_feature_importance=PLOT_FEATURE_IMPORTANCE)


utility.save_artifacts(logger, IS_TEST, PLOT_FEATURE_IMPORTANCE, 
                       result_dict, 
                       submission, 
                       MODEL_NUMBER, 
                       run_id, 
                       SUBMISSION_DIR, 
                       OOF_DIR, 
                       FI_DIR, 
                       FI_FIG_DIR)

end = timer()
utility.update_tracking(run_id, "training_time", (end - start), is_integer=True)
# Update the comments
utility.update_tracking(run_id, "comments", EXP_DETAILS)
logger.info('Done!')

