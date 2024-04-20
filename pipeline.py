import os
from typing import Dict
from pathlib import Path
import pandas as pd
import numpy as np

from datetime import datetime
from dateutil.relativedelta import relativedelta
import swifter


ROOT_DATA_PATH = Path('data')
NUM_WEEKS_BUFFER = 1

def initialize_paths():

    # create the root dir
    ROOT_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # create raw layer dir
    data_raw_output_path = Path(ROOT_DATA_PATH, 'raw')
    data_raw_output_path.mkdir(parents=True, exist_ok=True)

    # create silver layer dir
    data_silver_output_path = Path(ROOT_DATA_PATH, 'silver')
    data_silver_output_path.mkdir(parents=True, exist_ok=True)

    # create gold layer dir
    data_gold_output_path = Path(ROOT_DATA_PATH, 'gold')
    data_gold_output_path.mkdir(parents=True, exist_ok=True)

    return {'raw': data_raw_output_path,
            'silver': data_silver_output_path,
            'gold': data_gold_output_path}

def process_raw_data(data_raw_path: Path) -> None:
    """Reads and normalize data using Pandas
    """

    prints_raw_data = pd.read_json('prints.json', lines=True)
    taps_raw_data = pd.read_json('taps.json', lines=True)
    pays = pd.read_csv('pays.csv', sep=',')

    prints_raw_data.to_parquet(Path(data_raw_path, 'prints_raw.parquet'))
    taps_raw_data.to_parquet(Path(data_raw_path, 'taps_raw.parquet'))
    pays.to_parquet(Path(data_raw_path, 'pays.parquet'))

## Silver Layer Functions/Utils

# Functions preprocess normalize cols
def preprocessnormalize_cols(df: pd.DataFrame) -> pd.DataFrame:

    # process data types to future filtering
    df_process = df.copy()
    df_process['day'] = pd.to_datetime(df_process['day'])

    # normalize json columns
    df_process_norm = pd.concat([df_process,
                                 pd.json_normalize(df_process['event_data'])], axis=1)
    df_process_norm = df_process_norm.drop(columns=['event_data'])

    return  df_process_norm

def process_silver_layer_prints(local_files_config: Dict):

    # BASIC SILVER LAYER PROCESSING @PRINTS
    prints_raw_df = pd.read_parquet(Path(local_files_config['raw'], 'prints_raw.parquet'))
    prints_norm_df = preprocessnormalize_cols(df=prints_raw_df)
    prints_norm_df.to_parquet(Path(local_files_config['silver'], 'prints_silver.parquet'))

def process_silver_layer_taps(local_files_config: Dict):
    ## BASIC SILVER LAYER PROCESSING @taps
    taps_raw_df = pd.read_parquet(Path(local_files_config['raw'], 'taps_raw.parquet'))
    taps_norm_df = preprocessnormalize_cols(df=taps_raw_df)
    taps_norm_df['click_user_value_prop'] = True

    taps_norm_df.to_parquet(Path(local_files_config['silver'], 'taps_silver.parquet'))


def process_silver_layer_payments(local_files_config):

    # BASIC SILVER LAYER PROCESSING @payments
    payments_raw_df = pd.read_parquet(Path(local_files_config['raw'],'pays.parquet'))
    payments_raw_df['pay_date'] = pd.to_datetime(payments_raw_df['pay_date'])
    payments_raw_df.to_parquet(Path(local_files_config['silver'], 'payments_silver.parquet'))

    return payments_raw_df

def build_silver_layer(local_files_config):
    process_silver_layer_prints(local_files_config)
    process_silver_layer_taps(local_files_config)
    process_silver_layer_payments(local_files_config)

## Gold Layer Functions/Utils

def get_args_for_user_ts(prints_norm_df):

    # tomamos como referencia temporal el input del data set prints
    START_DAY = prints_norm_df['day'].min()
    END_DAY = prints_norm_df['day'].max()

    # generamos un inice de la serie para hacer el resampling de la serie de cada usuario
    ts_indexes = pd.date_range(start = START_DAY,end=END_DAY)
    df_augmt = pd.DataFrame(ts_indexes, columns=['day'])

    # 21 dias de lag para calcular el backfill
    args_agg_ts_user_level = (df_augmt, 21)

    return df_augmt, 21

def get_counts_user_level(df_user, df_augmt, week_buff=21):

    # get user level counts by value prop this functions serves for prints and taps
    df_user_temp = df_user.copy()
    df_user_temp = df_user_temp.set_index('day')\
                            .groupby('value_prop')\
                            .resample('D')\
                            .agg({'value_prop': ['count']})\
                            .unstack(0)

    # rename for readable cols
    df_user_temp.columns = ["_".join(list(col)).replace('__','')\
                            for col in df_user_temp.columns.tolist()]

    # augment hte TS to match the entire month (there might be a better way to do that)
    df_user_temp = df_augmt.merge(df_user_temp.reset_index(),
                                  on='day',
                                  how='left').set_index('day')

    # User a rolling object to build a cummulative sum with a window of 3 weeks
    df_user_temp = df_user_temp.fillna(0).rolling(
        week_buff,
        min_periods=1,
        closed='left').sum()

    return df_user_temp

def get_payments_user_level(df_payments_user, df_augmt, week_buff=21):

    df_user_temp = df_payments_user.copy()

    # get payments metrics daily using a grouped versiuon by value proposal
    # here count will retriebe the # of paymets and sum will retrieve the total amount
    df_user_temp = df_user_temp.set_index('pay_date')\
                            .groupby('value_prop')\
                            .resample('D')\
                            .agg({'total': ['sum', 'count']})\
                            .unstack(0)
    # rename columns and remove multi index
    df_user_temp.columns = ["_".join(list(col)).replace('__','')\
                            for col in df_user_temp.columns.tolist()]

    # augment time series to get an entiner month (there might be a better way to do that)
    df_user_temp = df_augmt.merge(df_user_temp.reset_index(),
                                  left_on='day',
                                  right_on='pay_date',
                                  how='left').set_index('pay_date')
    df_user_temp = df_user_temp.drop(columns=['day'])

    # User a rolling object to build a cummulative sum with a window of 3 weeks
    df_user_temp = df_user_temp.fillna(0).rolling(
        week_buff,
        min_periods=1,
        closed='left').sum()

    return df_user_temp

def get_joined_prints_report(prints_df: pd.DataFrame,
                             taps_df: pd.DataFrame,
                             prints_user_level_ts: pd.DataFrame,
                             taps_user_level_ts: pd.DataFrame,
                             payments_user_level_ts: pd.DataFrame):

    # JOINED TABLE @prints <-> @taps
    df_join_1 = prints_df.merge(right=taps_df,
                        on=['day','user_id','position'],
                        how="left",
                        validate='one_to_one',
                        suffixes=("_prints", "_taps"))
    
    check_df = df_join_1.loc[~df_join_1['click_user_value_prop'].isna()]
    assert (check_df['value_prop_prints']!=check_df['value_prop_taps']).sum()==0

    df_join_1 = df_join_1['click_user_value_prop'].fillna(False)

    # JOINED TABLE prints report <> @payments_last_3_weeks
    df_join_2 = df_join_1.merge(payments_user_level_ts,
                            left_on=['day','user_id'],
                            right_on=['pay_date','user_id'],
                            how='left')

    # JOINED TABLE prints report <> @user_views_last_3_weeks
    df_join_3 = df_join_2.merge(prints_user_level_ts,
                                left_on=['day','user_id'],
                                right_on=['day','user_id'],
                                how='left')

    # JOINED TABLE prints report <> @user_taps_last_3_weeks
    df_join_4 = df_join_3.merge(taps_user_level_ts,
                            left_on=['day','user_id'],
                            right_on=['day','user_id'],
                            how='left')

    return df_join_4

def build_gold_layer(local_files_config):

    prints_df = pd.read_parquet(Path(local_files_config['silver'], 'prints_silver.parquet'))
    taps_df = pd.read_parquet(Path(local_files_config['silver'], 'taps_silver.parquet'))
    payments_df = pd.read_parquet(Path(local_files_config['silver'], 'payments_silver.parquet'))

    # filter the last week
    cutoff_prints = (prints_df['day'].max() - relativedelta(weeks=NUM_WEEKS_BUFFER))\
                                                .strftime(format="%Y-%m-%d")

    # get params to build user level timeseries
    df_augmt, n_days = get_args_for_user_ts(prints_df)

    # build user-level time series
    prints_user_level_ts = prints_df.swifter\
                                    .groupby('user_id')\
                                    .apply(lambda x: get_counts_user_level(x, df_augmt, n_days))
    taps_user_level_ts = taps_df.swifter\
                                .groupby('user_id')\
                                .apply(lambda x: get_counts_user_level(x, df_augmt, n_days))

    # rename cols to readable names
    prints_user_level_ts.columns = [c.replace('_count_', '_user_views_')\
                                    for c in prints_user_level_ts.columns]
    taps_user_level_ts.columns = [c.replace('_count_', '_user_clicks_')\
                                  for c in taps_user_level_ts.columns]

    payments_user_level_ts = payments_df.swifter\
                                        .groupby('user_id')\
                                        .apply(
                            lambda x: get_payments_user_level(x, df_augmt, n_days))
    payments_user_level_ts = payments_user_level_ts.reset_index()
    # rename columns to readable names
    payments_user_level_ts.columns = [c.replace('total_', 'payment_')\
                                      for c in payments_user_level_ts.columns]
    payments_user_level_ts.columns = [c.replace('count_', 'qty_') \
                                      for c in payments_user_level_ts.columns]

    df_gold_layer = get_joined_prints_report(
        prints_df,
        taps_df,
        prints_user_level_ts,
        taps_user_level_ts,
        payments_user_level_ts)

    df_gold_layer.to_parquet(Path(local_files_config['gold'], 'joined_table_gold.parquet'))
    df_analysis = df_gold_layer.copy().loc[(df_gold_layer['day']>=cutoff_prints)]
    df_analysis.to_parquet(Path(local_files_config['gold'], 'analysis_table_processed.parquet'))

if __name__ == '__main__':

    local_files_config = initialize_paths()

    process_raw_data(local_files_config['raw'])
    build_silver_layer(local_files_config)
    build_gold_layer(local_files_config)


