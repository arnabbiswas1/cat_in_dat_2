import os
import pandas as pd

## CV Score : 0.7854
## LB Score : 0.58369

COMPETITION_NAME = 'cat-in-the-dat-ii'

SUBMISSION_DIR = '.'
SUBMISSION_FILE = 'sub_cat_baseline_with_ordered_ordinal_use_cat_feature_0123_1455_0.7854.csv'
IS_CLIP_NEEDED = True
SUBMISSION_NUMBER = 1
SUBMISSION_MESSAGE = '"Baseline with CatBoost manually ordered ord_1, ord_2 and then CatBoost default categorical feature handling is used"'

df = pd.read_csv(f'{SUBMISSION_DIR}/{SUBMISSION_FILE}')
print(df.head())
if IS_CLIP_NEEDED:
    df.loc[df.target < 0, 'target'] = 0

final_submission_file_name = f'{SUBMISSION_NUMBER}_{SUBMISSION_FILE}'
df.to_csv(f'{SUBMISSION_DIR}/{final_submission_file_name}', index=False)

submission_string = f'kaggle competitions  submit -c {COMPETITION_NAME} -f {SUBMISSION_DIR}/{final_submission_file_name} -m {SUBMISSION_MESSAGE}'

print(submission_string)

os.system(submission_string)

    