import pandas as pd
from pynytimes import NYTAPI
import datetime
import os
import warnings

NYT_OUTPUT_PATH = 'data/raw-nyt-data'
NYT_API_KEY = os.environ.get('NYT_API_KEY')
nytapi = NYTAPI(NYT_API_KEY, parse_dates=True)


def nyt_download_history(start_year=1900, output_folder=NYT_OUTPUT_PATH, overwrite=False, warn=False):
    # Note: API is limited to 500 requests per day, i.e. 500 months.
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
    os.makedirs(NYT_OUTPUT_PATH, exist_ok=True)
    last_file = os.listdir(NYT_OUTPUT_PATH)[-1]

    start_year = int(last_file[:4])
    start_month = int(last_file[5:7])

    end_year = datetime.datetime.now().year
    end_month = datetime.datetime.now().month

    for year in range(start_year, end_year + 1):
        start_month = 1 if year > start_year else start_month
        for month in range(start_month, 13):
            _download_nyt_headlines_for_month(month, year, output_folder, overwrite=True)
            if year == end_year and month == end_month:
                break


def _download_nyt_headlines_for_month(month, year, output_folder, overwrite=False):
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
    articles = nytapi.archive_metadata(date)
    headlines = []
    for article in articles:
        row = (article["pub_date"], article["headline"]["main"], article['news_desk'])
        headlines.append(row)
    return headlines


def __format_headlines(headlines):
    df = pd.DataFrame(headlines, columns=["date", "headline", "topic"])
    df = df.set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize(None)
    return df.sort_index()
