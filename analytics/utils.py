import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def read_file(path):
    df = pd.read_excel(path)
    df.columns = df.iloc[4]
    df = df.iloc[5:][df.columns[:-1]]

    columns = df.columns.tolist()

    columns[3] = 'продажи'
    columns[16] = 'заболеваемость'
    columns[33] = 'ТВ реклама, руб'
    columns[35] = 'интернет реклама, руб'
    columns[51] = 'wordstat'

    df.columns = columns
    df.loc[df['неделя']==53, 'неделя'] = 52

    return df



def create_features(df):

    week_info = df.groupby('неделя')['продажи'].agg(['sum', 'count'])
    week_info.columns = [f'week_{x}' for x in week_info.columns]

    # df = df.merge(week_info, how='left', right_index=True, left_on='неделя')
    # df['week_sum'] -= df['продажи']
    # df['week_count'] -= 1
    # df['feature_mean_week'] = df['week_sum'] / df['week_count']
    # df['feature_mean_week_norm'] = df['feature_mean_week'] / df['продажи'].shift(1)
    # df.drop(['week_sum', 'week_count'], axis=1, inplace=True)

    for i in range(29):
        df[f'target_{i+1}'] = df['продажи'].shift(-i) / df['продажи'].shift(1)

    # week_info_dict = week_info.to_dict()
    for i in range(29):
        df['tmp'] = df['неделя'] + i
        df = df.merge(week_info, how='left', right_index=True, left_on='tmp')
        df['week_sum'] -= df['продажи'].shift(-i)
        df['week_count'] -= 1
        df[f'feature_mean_week_{i}'] = df['week_sum'] / df['week_count'] / 10_000_000
        df[f'feature_mean_week_norm_{i}'] = df[f'feature_mean_week_{i}'] / df['продажи'].shift(1) / 10_000_000
        df.drop(['week_sum', 'week_count'], axis=1, inplace=True)

    df['feature_illnesses'] = df['заболеваемость'].shift(1) / 1_000_000

    df['feature_week'] = df['неделя']

    # for i in range(1, 16, 4):
    #     df[f'feature_month_diff_{i}'] = df['продажи'].shift(i) / df['продажи'].shift(i+4) / 10_000_000

    for i in [7, 13, 20]:
        df[f'feature_big_diff_{i}_weeks'] = df['продажи'].shift(1) / df['продажи'].shift(i) / 10_000_000

    # bad_cols = [col for col in df.columns if 'feature' not in col and 'target' not in col]
    # df.drop(bad_cols, axis=1, inplace=True)
    
    return df


def split_df(cur_df, i):
    test_size = 0.2
    cur_df = cur_df[cur_df[f'target_{i}'].notna()]
    cur_df = cur_df.drop([43-i, 44-i, 45-i, 46-i])
    train_data, valid_data = cur_df[:int(len(cur_df) * (1-test_size))], cur_df[int(len(cur_df) * (1-test_size)):]

    X_train, y_train = train_data.drop([x for x in train_data.columns if 'target' in x], axis=1), train_data[f'target_{i}']
    X_valid, y_valid = valid_data.drop([x for x in valid_data.columns if 'target' in x], axis=1), valid_data[f'target_{i}']


def create_illness_features(cur_df):
    df = cur_df.copy(deep=True)

    week_info = df.groupby('неделя')['заболеваемость'].agg(['sum', 'count'])
    week_info.columns = [f'week_{x}' for x in week_info.columns]

    df = df.merge(week_info, how='left', right_index=True, left_on='неделя')
    df['week_sum'] -= df['заболеваемость']
    df['week_count'] -= 1
    df['feature_mean_week'] = df['week_sum'] / df['week_count']
    df['feature_mean_week_norm'] = df['feature_mean_week'] / df['заболеваемость'].shift(1)

    df['feature_week'] = df['неделя']

    for i in range(28):
        df[f'target_{i+1}'] = df['заболеваемость'].shift(-i) / df['заболеваемость'].shift(1)

    for i in [7, 13, 20]:
        df[f'feature_big_diff_{i}_weeks'] = df['заболеваемость'].shift(1) / df['заболеваемость'].shift(i)

    # bad_cols = [col for col in df.columns if 'feature' not in col and 'target' not in col]
    # df.drop(bad_cols, axis=1, inplace=True)
    
    return df