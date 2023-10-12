import os
import pandas as pd


def create_sentiment_index(
    input_folder='data/classified-nyt-data',
    output_path='data/nyt-index.csv'
):
    """Create sentiment index file using classified headlines from input folder.

    Parameters
    ----------
    input_folder : str, optional
        Path to the folder containing CSV files of classified NYT data, by
        default 'data/classified-nyt-data'.
    output_path : str, optional
        Path where the output CSV, Feather, or Parquet file will be stored, by default
        'data/nyt-index.csv'.

    Returns
    -------
    pd.DataFrame
        The index as a pandas DataFrame.
    """
    create_index_file(output_path)

    for f in sorted(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, f)
        classified_headlines = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = convert_headlines_df_to_index(classified_headlines)
        if df is not None:
            _append_index_to_file(df, output_path)

    index_df = add_smoothed_col_to_index_df(output_path)
    _write_index_to_file(index_df, output_path)
    return index_df


def append_sentiment_index(
    input_folder='data/classified-nyt-data',
    output_path='data/nyt-index.csv'
):
    """Append new data to the sentiment index file from the given input folder.

    Parameters
    ----------
    input_folder : str, optional
        Path to the folder containing CSV files of classified NYT data, by
        default 'data/classified-nyt-data'.
    output_path : str, optional
        Path where the output CSV, Feather, or Parquet file will be stored, by default
        'data/nyt-index.csv'.

    Returns
    -------
    pd.DataFrame
        The index as a pandas DataFrame.
    """
    index_df = _read_index_file(output_path)
    index_df = index_df[['negative', 'neutral', 'positive', 'total', 'index_value']]
    append_start_date = index_df.index[-1] + pd.DateOffset(days=1)

    for f in files_missing_in_index(input_folder, append_start_date):
        file_path = os.path.join(input_folder, f)
        classified_headlines = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df = convert_headlines_df_to_index(classified_headlines)
        if isinstance(df, (pd.DataFrame, pd.Series)):
            df = df.loc[append_start_date:]
            df['smoothed_index_value'] = 0
            _append_index_to_file(df, output_path)

    index_df = add_smoothed_col_to_index_df(output_path)
    _write_index_to_file(index_df, output_path)
    return index_df


def _read_index_file(index_path):
    """Read the index from a CSV, Feather or Parquet file.

    Parameters
    ----------
    index_path : str
        The path to the input file, must end with .csv, .feather, .pq or .parquet.

    Returns
    -------
    DataFrame
        The index as a pandas DataFrame.
    """
    if index_path.endswith('.csv'):
        index_df = pd.read_csv(index_path, index_col=0, parse_dates=True)
    elif index_path.endswith('.feather'):
        index_df = pd.read_feather(index_path)
        index_df = index_df.set_index('date')
    elif index_path.endswith('.parquet') or index_path.endswith('.pq'):
        index_df = pd.read_parquet(index_path)
    else:
        raise ValueError('Index file must be CSV, Feather or Parquet')
    index_df.index = pd.DatetimeIndex(index_df.index)
    return index_df


def _write_index_to_file(index_df, output_path):
    """Write the index DataFrame to a CSV, Feather or Parquet file.

    Parameters
    ----------
    index_df : pandas.DataFrame
        The DataFrame to write to file.
    output_path : str
        The path to the output file, must end with .csv, .feather, .pq or .parquet.
    """
    if output_path.endswith('.csv'):
        index_df.to_csv(output_path)
    elif output_path.endswith('.feather'):
        index_df = index_df.reset_index()
        index_df.to_feather(output_path)
    elif output_path.endswith('.parquet') or output_path.endswith('.pq'):
        index_df.to_parquet(output_path)
    else:
        raise ValueError('Output file must be CSV, Feather or Parquet')


def _append_index_to_file(df, output_path):
    """Append the index DataFrame to a CSV, Feather or Parquet file.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to append to file.
    output_path : str
        The path to the output file, must end with .csv.
    """
    if output_path.endswith('.csv'):
        df.to_csv(output_path, header=False, mode='a')
    else:
        old_df = _read_index_file(output_path)
        if not old_df.empty:
            df = pd.concat([old_df, df])
        _write_index_to_file(df, output_path)


def create_index_file(file_path):
    """Create an empty index file with specified column names.

    Parameters
    ----------
    file_path : str
        Path where the output file will be stored.
    """
    COLUMN_NAMES = ['date', 'negative', 'neutral', 'positive',
                    'total', 'index_value']
    df = pd.DataFrame(columns=COLUMN_NAMES)
    df.set_index('date', inplace=True)
    if file_path.endswith('.csv'):
        df.to_csv(file_path, index=False)
    elif file_path.endswith('.feather'):
        df = df.reset_index()
        df.to_feather(file_path)
    elif file_path.endswith('.parquet') or file_path.endswith('.pq'):
        df.to_parquet(file_path)


def files_missing_in_index(input_folder, append_start_date):
    """Return the files that are missing from the index.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the input CSV files.
    append_start_date : datetime
        The starting date for the files that will be appended.

    Returns
    -------
    list
        List of file names that are missing in the index.
    """
    files = sorted(os.listdir(input_folder))
    files = [f for f in files if f.endswith(".csv")]
    month_start = append_start_date.replace(day=1)
    files = [f for f in files if _file_to_dtime(f) >= month_start]
    return files


def _file_to_dtime(file_name):
    """Convert the date string in a file name to a datetime object.
    The datetime object is the first day of the month given in the file name.

    Parameters
    ----------
    file_name : str
        The name of the file to be converted.

    Returns
    -------
    datetime
        The datetime object corresponding to the file name.
    """
    date_str = file_name.split('.')[0]
    return pd.to_datetime(date_str)


def convert_headlines_df_to_index(classified_headlines):
    """Convert a DataFrame of classified headlines into an index DataFrame.

    Parameters
    ----------
    classified_headlines : pd.DataFrame
        DataFrame of headlines with sentiment and topic labels.

    Returns
    -------
    DataFrame or None
        Formatted DataFrame if successful, None if an error occurs.
    """
    try:
        classified_headlines = _filter_headlines(classified_headlines)
        df = _format_to_index(classified_headlines)
        df = _drop_today_if_in_df(df)  # To avoid writing incomplete data
        return df
    except AttributeError:  # If file is empty
        return None


def _filter_headlines(df):
    """Filter DataFrame for 'Economics' headlines longer than 2 words.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be filtered.

    Returns
    -------
    DataFrame
        The filtered DataFrame.
    """
    df = df[df['model topic'] == 'Economics']
    # drop headlines with less than 3 words
    df = df[df['headline'].str.split().str.len() >= 3]
    return df


def _format_to_index(df):
    """Formats DataFrame of classified NYT data into the index format.

    Counts the number of negative, neutral and positive sentiment labels per day and
    calculates the index value and total number of headlines per day.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be formatted.

    Returns
    -------
    DataFrame
        The formatted DataFrame.
    """
    df = __count_num_labels_per_day(df)
    df['total'] = df.sum(axis=1)
    df['index_value'] = (
        (df['positive'] - df['negative']) / (df['positive'] + df['negative'])
    )
    df = __resample_to_day(df, index_col='index_value')
    return df


def __count_num_labels_per_day(df):
    """Count the number of negative, neutral, and positive sentiment labels per day.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with a 'sentiment' column and datetime index.

    Returns
    -------
    DataFrame
        DataFrame with date as index and counts of negative, neutral, and positive
        labels as columns.
    """
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
    """Resample df to daily freq and fill missing index values with rolling mean.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be resampled.
    index_col : str
        Name of the column to use for filling missing values.

    Returns
    -------
    DataFrame
        The resampled DataFrame.
    """
    df.index = pd.DatetimeIndex(df.index)
    sma = df[index_col].rolling(365, min_periods=1).mean()
    sma = sma.resample('D').ffill()

    df = df.resample('D').mean()
    df[index_col] = df[index_col].fillna(sma)
    df = df.fillna(0)
    df.index.name = 'date'
    return df


def _drop_today_if_in_df(df):
    """Drop today's data from the DataFrame if it exists.

    Parameters
    ----------
    df : DataFrame
        The DataFrame from which today's data
    should be dropped if it exists.

    Returns
    -------
    DataFrame
        The DataFrame with today's data possibly dropped.
    """
    today = pd.Timestamp.today().floor('D')
    latest_day_is_today = df.index[-1] == today
    if latest_day_is_today:
        df = df.iloc[:-1]
    return df


def add_smoothed_col_to_index_df(index_path):
    """Read index file and add a column of smoothed index values.

    Parameters
    ----------
    index_path : str
        Path to the index CSV file.

    Returns
    -------
    DataFrame
        DataFrame with an added column of smoothed index values.
    """
    index_df = _read_index_file(index_path)
    index_df = index_df[['negative', 'neutral', 'positive', 'total', 'index_value']]
    smoothed_index = __calculate_smoothed_index(index_df['index_value'])
    index_df['smoothed_index_value'] = smoothed_index
    index_df.index.name = 'date'
    return index_df


def __calculate_smoothed_index(index_series):
    """Calculate a smoothed and detrended index value from the given series.

    Uses 100-day EMA as smoothing and 7-year SMA as trend.

    Parameters
    ----------
    index_series : Series
        The series from which to calculate the smoothed and detrended index.

    Returns
    -------
    Series
        The series with the calculated index values.
    """
    SEVEN_YEARS = 365 * 7
    smoothed_df = index_series.ewm(span=100).mean()
    trend_df = index_series.rolling(SEVEN_YEARS).mean()
    index = smoothed_df - trend_df + 0.5  # Detrended index
    return index
