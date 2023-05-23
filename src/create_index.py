import os
import pandas as pd


def create_index(input_folder='data/classified-nyt-data', output_path='data/nyt-index.csv'):
    COLUMN_NAMES = ['date', 'negative', 'neutral', 'positive',
                    'total', 'index_value']
    pd.DataFrame(columns=COLUMN_NAMES).to_csv(output_path, index=False)
    for f in os.listdir(input_folder):
        if f.endswith(".csv"):
            df = pd.read_csv(
                os.path.join(input_folder, f), index_col=0,
                parse_dates=True, na_values=['None']
            )
            df = _filter_df(df)
            try:
                df = _format_df(df)
                df = _drop_today_if_in_df(df)  # To avoid writing incomplete data
                df.to_csv(output_path, header=False, mode='a')
            except AttributeError as e:
                print(f'Error in file {f}: {e}')
                continue

    index = pd.read_csv(output_path, index_col=0, parse_dates=True)
    index['smoothed_index_value'] = _add_smoothed_index_value(index['index_value'])
    index.to_csv(output_path)


def append_index(input_folder='data/classified-nyt-data', output_path='data/nyt-index.csv'):
    index = pd.read_csv(output_path, index_col=0, parse_dates=True)
    index = index[['negative', 'neutral', 'positive', 'total', 'index_value']]
    append_start_date = index.index[-1] + pd.DateOffset(days=1)

    for f in _files_missing_in_index(input_folder, append_start_date):
        try:
            df = pd.read_csv(os.path.join(input_folder, f), index_col=0, parse_dates=True)
            df = _filter_df(df)
            df = _format_df(df)
            df = df.loc[append_start_date:]
            df = _drop_today_if_in_df(df)
            index = pd.concat([index, df], axis=0)
        except AttributeError as e:
            print(f'Error in file {f}: {e}')
            continue

    index['smoothed_index_value'] = _add_smoothed_index_value(index['index_value'])
    index.index.name = 'date'
    index.to_csv(output_path)


def _files_missing_in_index(input_folder, append_start_date):
    files = os.listdir(input_folder)
    files = [f for f in files if f.endswith(".csv")]
    month_start = append_start_date.replace(day=1)
    files = [f for f in files if __file_to_dtime(f) >= month_start]
    return files


def __file_to_dtime(file_name):
    """Convert date string to month and year"""
    date_str = file_name.split('.')[0]
    return pd.to_datetime(date_str)


def _filter_df(df):
    df = df[df['model topic'] == 'Economics']
    # drop headlines with less than 3 words
    df = df[df['headline'].str.split().str.len() >= 3]
    return df


def _format_df(df):
    df.index.name = 'date'
    df = __count_num_labels_per_day(df)
    df['total'] = df.sum(axis=1)
    df['index_value'] = (df['positive'] - df['negative']) / (df['positive'] + df['negative'])
    df = __resample_to_day(df, index_col='index_value')
    return df


def __count_num_labels_per_day(df):
    """Count number of negative, neutral and positive sentiment labels per day"""
    df = df['sentiment'].groupby(df.index.date).value_counts().unstack()
    df = df.fillna(0)
    for col in ['Negative', 'Neutral', 'Positive']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].astype(int)
    df = df[['Negative', 'Neutral', 'Positive']]
    df.columns = ['negative', 'neutral', 'positive']
    return df.sort_index()


def __resample_to_day(df, index_col):
    """Resample to daily frequency and fill missing index values with rolling mean"""
    df.index = pd.DatetimeIndex(df.index)
    sma = df[index_col].rolling(365, min_periods=1).mean()
    sma = sma.resample('D').ffill()

    df = df.resample('D').mean()
    df[index_col] = df[index_col].fillna(sma)
    df = df.fillna(0)
    return df


def _add_smoothed_index_value(index_series):
    """Calculate smoothed and detrended index value.
    Uses 100-day EMA as smoothing and 10-year SMA as trend.
    """
    TEN_YEARS = 365 * 10
    smoothed_df = index_series.ewm(span=100).mean()
    trend_df = index_series.rolling(TEN_YEARS).mean()
    index = smoothed_df - trend_df + 0.5  # Detrended index
    return index


def _drop_today_if_in_df(df):
    """Drop today's data if it is in the dataframe.
    This is to avoid having incomplete data in the index.
    """
    today = pd.Timestamp.today().floor('D')
    latest_day_is_today = df.index[-1] == today
    if latest_day_is_today:
        df = df.iloc[:-1]
    return df
