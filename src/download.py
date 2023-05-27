import os
import datetime
import warnings

import pandas as pd
from pynytimes import NYTAPI

NYT_OUTPUT_PATH = 'data/raw-nyt-data'
NYT_API_KEY = os.environ.get('NYT_API_KEY')
try:
    nytapi = NYTAPI(NYT_API_KEY, parse_dates=True)
except ValueError:
    raise ValueError('NYT_API_KEY environment variable not set. You can get '
                     'an API-key from https://developer.nytimes.com.')


def nyt_download_history(
    start_year=1900,
    output_folder=NYT_OUTPUT_PATH,
    overwrite=False,
    warn=False
):
    """Download historical New York Times headlines starting from a specified year.

    Note: API is limited to 500 requests per day, i.e. 500 months.

    Parameters
    ----------
    start_year : int, optional
        The year from which to start downloading headlines. Default is 1900.
    output_folder : str, optional
        The path to the folder where the downloaded files will be saved.
        Default is 'data/raw-nyt-data'.
    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
    warn : bool, optional
        If True, issue a warning when a file already exists and is not overwritten.
        Default is False.
    """
    os.makedirs(NYT_OUTPUT_PATH, exist_ok=True)
    end_year = datetime.datetime.now().year
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if datetime.datetime(year, month, 1) > datetime.datetime.now():
                break
            try:
                _download_nyt_headlines_for_month(
                    month, year, output_folder, overwrite=overwrite
                )
            except FileExistsError:
                if warn:
                    warnings.warn(f'{year}-{month:02d} already downloaded, skipping...')


def nyt_download_latest(output_folder=NYT_OUTPUT_PATH):
    """Download the latest historical New York Times headlines.

    Note: API is limited to 500 requests per day, i.e. 500 months.

    Parameters
    ----------
    output_folder : str, optional
        The path to the folder where the downloaded files will be saved.
        Default is 'data/raw-nyt-data'.
    """
    last_file = os.listdir(NYT_OUTPUT_PATH)[-1]

    start_year = int(last_file[:4])
    start_month = int(last_file[5:7])

    end_year = datetime.datetime.now().year
    end_month = datetime.datetime.now().month

    for year in range(start_year, end_year + 1):
        start_month = 1 if year > start_year else start_month
        for month in range(start_month, 13):
            _download_nyt_headlines_for_month(
                month, year, output_folder, overwrite=True
            )
            if year == end_year and month == end_month:
                break


def _download_nyt_headlines_for_month(month, year, output_folder, overwrite=False):
    """Download New York Times headlines for a specified month and year.

    Parameters
    ----------
    month : int
        The month for which to download headlines.
    year : int
        The year for which to download headlines.
    output_folder : str
        The path to the folder where the downloaded files will be saved.
    overwrite : bool, optional
        If True, overwrite existing files. Default is False.
    """
    output_path = f'{output_folder}/{year}-{month:02d}.csv'
    if not overwrite and os.path.exists(output_path):
        raise FileExistsError(f'File {output_path} already exists')

    date = datetime.datetime(year, month, 1)
    if date > datetime.datetime.now():
        raise ValueError('Cannot download headlines for future dates')

    headlines = __get_nyt_headlines(date)
    df = __format_headlines(headlines)
    df.to_csv(output_path)


def __get_nyt_headlines(date):
    """Get New York Times headlines for a specified month.

    Parameters
    ----------
    date : datetime
        The month for which to get headlines.

    Returns
    -------
    list
        A list of tuples, each containing the publication date, headline, and
        news desk of an article.
    """
    articles = nytapi.archive_metadata(date)
    headlines = []
    for article in articles:
        row = (
            article["pub_date"],
            article["headline"]["main"],
            article['news_desk'],
        )
        headlines.append(row)
    return headlines


def __format_headlines(headlines):
    """Format headlines data into a DataFrame.

    Parameters
    ----------
    headlines : list
        A list of tuples, each containing the publication date, headline,
        and news desk of an article.

    Returns
    -------
    DataFrame
        A DataFrame containing the formatted headlines data.
    """
    df = pd.DataFrame(headlines, columns=["date", "headline", "topic"])
    df = df.set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize(None)
    return df.sort_index()
