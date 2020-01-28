import os
import pandas as pd

## CV Score : 0.7854
## LB Score : 0.78435

"""
Same as submission 1, but removed the clipping (negative probabilities to zero for target).
Looks like that impacted the previous score
"""


COMPETITION_NAME = 'cat-in-the-dat-ii'

SUBMISSION_DIR = '.'
SUBMISSION_FILE = 'sub_cat_baseline_with_ordered_ordinal_use_cat_feature_0123_1455_0.7854.csv'
IS_CLIP_NEEDED = False
SUBMISSION_NUMBER = 2
SUBMISSION_MESSAGE = '"No clipping on top of submission 1"'

df = pd.read_csv(f'{SUBMISSION_DIR}/{SUBMISSION_FILE}')
print(df.head())
if IS_CLIP_NEEDED:
    df.loc[df.target < 0, 'target'] = 0

final_submission_file_name = f'{SUBMISSION_NUMBER}_{SUBMISSION_FILE}'
df.to_csv(f'{SUBMISSION_DIR}/{final_submission_file_name}', index=False)

submission_string = f'kaggle competitions  submit -c {COMPETITION_NAME} -f {SUBMISSION_DIR}/{final_submission_file_name} -m {SUBMISSION_MESSAGE}'

print(submission_string)

os.system(submission_string)

    